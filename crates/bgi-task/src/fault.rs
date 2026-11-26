use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaultBoundary {
    Capture,
    Minimap,
    BigMap,
    Ocr,
    Motion,
    Input,
    Cancellation,
}

#[derive(Debug, Clone)]
pub struct Fault {
    pub boundary: FaultBoundary,
    pub at_call: u64,
    pub message: String,
}

#[derive(Debug, Default)]
struct FaultState {
    faults: BTreeMap<(FaultBoundary, u64), String>,
    calls: BTreeMap<FaultBoundary, u64>,
}

#[derive(Debug, Clone, Default)]
pub struct FaultInjector {
    state: Arc<Mutex<FaultState>>,
}

impl FaultInjector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&self, boundary: FaultBoundary, at_call: u64, message: impl Into<String>) {
        assert!(at_call > 0, "fault call numbers are one-based");
        self.state
            .lock()
            .unwrap()
            .faults
            .insert((boundary, at_call), message.into());
    }

    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap();
        state.faults.clear();
        state.calls.clear();
    }

    pub fn check(&self, boundary: FaultBoundary) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        let call = state.calls.entry(boundary).or_default();
        *call = call.saturating_add(1);
        let call_number = *call;
        if let Some(message) = state.faults.remove(&(boundary, call_number)) {
            Err(message)
        } else {
            Ok(())
        }
    }

    pub fn call_count(&self, boundary: FaultBoundary) -> u64 {
        self.state
            .lock()
            .unwrap()
            .calls
            .get(&boundary)
            .copied()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injection_is_one_shot_and_scoped_to_boundary() {
        let injector = FaultInjector::new();
        injector.add(FaultBoundary::Ocr, 2, "ocr unavailable");
        assert!(injector.check(FaultBoundary::Capture).is_ok());
        assert!(injector.check(FaultBoundary::Ocr).is_ok());
        assert_eq!(
            injector.check(FaultBoundary::Ocr),
            Err("ocr unavailable".into())
        );
        assert!(injector.check(FaultBoundary::Ocr).is_ok());
    }
}
