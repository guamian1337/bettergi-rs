use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

#[derive(Debug)]
pub(crate) struct ActiveTask {
    pub id: u64,
    pub cancel: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
    release_inputs_on_stop: bool,
}

impl ActiveTask {
    pub fn new(id: u64, cancel: Arc<AtomicBool>, join: JoinHandle<()>) -> Self {
        Self::with_input_policy(id, cancel, join, true)
    }

    pub fn with_input_policy(
        id: u64,
        cancel: Arc<AtomicBool>,
        join: JoinHandle<()>,
        release_inputs_on_stop: bool,
    ) -> Self {
        Self {
            id,
            cancel,
            join: Some(join),
            release_inputs_on_stop,
        }
    }

    pub const fn releases_inputs_on_stop(&self) -> bool {
        self.release_inputs_on_stop
    }

    pub fn request_stop(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub fn accepts(&self, task_id: u64) -> bool {
        self.id == task_id
    }

    pub fn take_join(&mut self) -> Option<JoinHandle<()>> {
        self.join.take()
    }
}

pub(crate) fn allocate_task_id(next_id: &mut u64) -> u64 {
    let id = *next_id;
    *next_id = next_id.checked_add(1).expect("task id overflow");
    id
}

pub(crate) fn accepts_event(active: &Option<ActiveTask>, task_id: u64) -> bool {
    active.as_ref().is_some_and(|task| task.accepts(task_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_ids_are_monotonic() {
        let mut next = 1;
        assert_eq!(allocate_task_id(&mut next), 1);
        assert_eq!(allocate_task_id(&mut next), 2);
    }

    #[test]
    fn observed_tasks_can_disable_stop_input_without_changing_defaults() {
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let join = std::thread::spawn(move || {
            while !worker_cancel.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        });
        let active = ActiveTask::with_input_policy(8, cancel, join, false);
        assert!(!active.releases_inputs_on_stop());
        let mut completed = active;
        completed.request_stop();
        completed.take_join().unwrap().join().unwrap();
    }

    #[test]
    fn stale_events_are_rejected_and_stop_keeps_lease_until_exit() {
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let join = std::thread::spawn(move || {
            while !worker_cancel.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        });
        let mut active = Some(ActiveTask::new(7, cancel.clone(), join));

        assert!(accepts_event(&active, 7));
        assert!(!accepts_event(&active, 6));
        active.as_ref().unwrap().request_stop();
        assert!(cancel.load(Ordering::Acquire));
        assert!(accepts_event(&active, 7));
        let mut completed = active.take().unwrap();
        completed.take_join().unwrap().join().unwrap();
        assert!(!accepts_event(&active, 7));
    }
}
