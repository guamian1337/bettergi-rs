use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::{Generation, RuntimeError};

#[derive(Debug, Clone)]
struct LeaseEntry {
    id: u64,
    generation: Generation,
    name: String,
    options: Value,
}

#[derive(Debug, Default)]
struct LeaseState {
    next_id: u64,
    generation: Generation,
    entries: Vec<LeaseEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriggerLeaseSnapshot {
    pub id: u64,
    pub name: String,
    pub options: Value,
}

#[derive(Debug, Clone, Default)]
pub struct TriggerLeaseRegistry {
    inner: Arc<Mutex<LeaseState>>,
}

impl TriggerLeaseRegistry {
    pub fn lease(
        &self,
        name: impl Into<String>,
        options: Value,
    ) -> Result<TriggerLease, RuntimeError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RuntimeError::InvalidLeaseName);
        }
        let mut state = self.inner.lock().unwrap();
        state.next_id = state
            .next_id
            .checked_add(1)
            .ok_or(RuntimeError::LeaseOverflow)?;
        let id = state.next_id;
        let generation = state.generation;
        state.entries.push(LeaseEntry {
            id,
            generation,
            name: name.clone(),
            options,
        });
        Ok(TriggerLease {
            registry: self.clone(),
            id,
            generation,
            name,
        })
    }

    pub fn snapshot(&self) -> Vec<TriggerLeaseSnapshot> {
        self.inner
            .lock()
            .unwrap()
            .entries
            .iter()
            .map(|entry| TriggerLeaseSnapshot {
                id: entry.id,
                name: entry.name.clone(),
                options: entry.options.clone(),
            })
            .collect()
    }

    pub fn clear(&self) {
        let mut state = self.inner.lock().unwrap();
        state.entries.clear();
        state.generation.0 = state
            .generation
            .0
            .checked_add(1)
            .ok_or(RuntimeError::LeaseOverflow)
            .unwrap();
    }

    fn dispose(&self, id: u64, generation: Generation) {
        let mut state = self.inner.lock().unwrap();
        if state.generation != generation {
            return;
        }
        state
            .entries
            .retain(|entry| entry.id != id || entry.generation != generation);
    }
}

#[derive(Debug)]
pub struct TriggerLease {
    registry: TriggerLeaseRegistry,
    id: u64,
    generation: Generation,
    name: String,
}

impl TriggerLease {
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for TriggerLease {
    fn drop(&mut self) {
        self.registry.dispose(self.id, self.generation);
    }
}

#[derive(Debug, Default)]
struct PauseState {
    generation: Generation,
    next_owner: u64,
    active_owners: BTreeSet<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct PauseController {
    inner: Arc<Mutex<PauseState>>,
}

impl PauseController {
    pub fn pause(&self) -> Result<PauseLease, RuntimeError> {
        let mut state = self.inner.lock().unwrap();
        let owner = state
            .next_owner
            .checked_add(1)
            .ok_or(RuntimeError::LeaseOverflow)?;
        state.next_owner = owner;
        state.active_owners.insert(owner);
        Ok(PauseLease {
            controller: self.clone(),
            owner,
            generation: state.generation,
        })
    }

    pub fn try_pause(&self) -> Result<Option<PauseLease>, RuntimeError> {
        let mut state = self.inner.lock().unwrap();
        if !state.active_owners.is_empty() {
            return Ok(None);
        }
        let owner = state
            .next_owner
            .checked_add(1)
            .ok_or(RuntimeError::LeaseOverflow)?;
        state.next_owner = owner;
        state.active_owners.insert(owner);
        Ok(Some(PauseLease {
            controller: self.clone(),
            owner,
            generation: state.generation,
        }))
    }

    pub fn is_paused(&self) -> bool {
        !self.inner.lock().unwrap().active_owners.is_empty()
    }

    pub fn resume_all(&self) {
        let mut state = self.inner.lock().unwrap();
        state.generation.0 = state
            .generation
            .0
            .checked_add(1)
            .expect("pause generation overflow");
        state.active_owners.clear();
    }

    fn release(&self, owner: u64, generation: Generation) {
        let mut state = self.inner.lock().unwrap();
        if state.generation == generation {
            state.active_owners.remove(&owner);
        }
    }
}

#[derive(Debug)]
pub struct PauseLease {
    controller: PauseController,
    owner: u64,
    generation: Generation,
}

impl PauseLease {
    pub fn is_active(&self) -> bool {
        let state = self.controller.inner.lock().unwrap();
        state.generation == self.generation && state.active_owners.contains(&self.owner)
    }
}

impl Drop for PauseLease {
    fn drop(&mut self) {
        self.controller.release(self.owner, self.generation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn trigger_lease_is_idempotent_and_last_entry_wins() {
        let registry = TriggerLeaseRegistry::default();
        let first = registry.lease("pick", json!({"mode": "a"})).unwrap();
        let second = registry.lease("pick", json!({"mode": "b"})).unwrap();
        assert_eq!(
            registry.snapshot().last().unwrap().options,
            json!({"mode": "b"})
        );
        drop(second);
        assert_eq!(
            registry.snapshot().last().unwrap().options,
            json!({"mode": "a"})
        );
        drop(first);
        assert!(registry.snapshot().is_empty());
    }

    #[test]
    fn exclusive_pause_rejects_busy_input_and_tracks_invalidation() {
        let controller = PauseController::default();
        let first = controller.try_pause().unwrap().unwrap();
        assert!(controller.try_pause().unwrap().is_none());
        controller.resume_all();
        assert!(!first.is_active());
        let second = controller.try_pause().unwrap().unwrap();
        drop(first);
        assert!(second.is_active());
        drop(second);
        assert!(controller.try_pause().unwrap().is_some());
    }

    #[test]
    fn stale_pause_lease_cannot_resume_new_generation() {
        let controller = PauseController::default();
        let stale = controller.pause().unwrap();
        controller.resume_all();
        let current = controller.pause().unwrap();
        drop(stale);
        assert!(controller.is_paused());
        drop(current);
        assert!(!controller.is_paused());
    }
}
