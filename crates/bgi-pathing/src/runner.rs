use std::time::{Duration, Instant};

use bgi_control::{TaskOutcome, TaskReport};
use bgi_task::AutoPickConfig;

use crate::PathingTaskFile;
use crate::executor::{PathExecutor, PathingHost};

#[derive(Debug, Clone)]
pub struct LoopPolicy {
    pub runs: u32,
    pub delay: Duration,
    pub wait_for_loading: bool,
    pub auto_pick: bool,
    pub auto_pick_config: Option<AutoPickConfig>,
    pub loading_timeout: Duration,
}

impl Default for LoopPolicy {
    fn default() -> Self {
        Self {
            runs: 1,
            delay: Duration::from_secs(1),
            wait_for_loading: false,
            auto_pick: false,
            auto_pick_config: None,
            loading_timeout: Duration::from_secs(60),
        }
    }
}

pub struct LoopRunner;

impl LoopRunner {
    pub fn run<H: PathingHost + ?Sized>(
        host: &mut H,
        task_id: u64,
        task_name: &str,
        task: &PathingTaskFile,
        policy: LoopPolicy,
    ) -> TaskReport {
        let started = Instant::now();
        let mut report = TaskReport {
            task_id,
            task_name: task_name.to_string(),
            outcome: TaskOutcome::Success,
            runs_completed: 0,
            runs_requested: policy.runs,
            current_index: None,
            total_points: task.positions.len(),
            failure: None,
            stage: "startup".into(),
            elapsed_ms: 0,
        };
        let mut pick_matcher = bgi_task::AutoPickMatcher::new();
        let mut pick_gate = bgi_task::AutoPickPromptGate::default();
        loop {
            if host.is_cancelled() {
                report.current_index = host.current_waypoint_index();
                return Self::finish_cancelled(&mut report, started);
            }
            if policy.wait_for_loading {
                report.stage = "loading".into();
                if host.is_cancelled() {
                    report.current_index = host.current_waypoint_index();
                    return Self::finish_cancelled(&mut report, started);
                }
                if let Err(error) = Self::wait_for_loading(host, policy.loading_timeout) {
                    if host.is_cancelled() {
                        report.current_index = host.current_waypoint_index();
                        return Self::finish_cancelled(&mut report, started);
                    }
                    return Self::finish_error(&mut report, started, error);
                }
            }
            report.stage = "pathing".into();
            if let Err(error) = PathExecutor::run(host, task_name, task) {
                report.current_index = host.current_waypoint_index();
                return Self::finish_error(&mut report, started, error);
            }
            if policy.auto_pick
                && let Some(config) = &policy.auto_pick_config
            {
                report.stage = "auto_pick".into();
                for _ in 0..3 {
                    if host.is_cancelled() {
                        report.current_index = host.current_waypoint_index();
                        return Self::finish_cancelled(&mut report, started);
                    }
                    if let Err(error) =
                        crate::auto_pick::pick_once(host, &mut pick_matcher, &mut pick_gate, config)
                    {
                        report.current_index = host.current_waypoint_index();
                        return Self::finish_error(&mut report, started, error);
                    }
                }
            }
            report.runs_completed = report.runs_completed.saturating_add(1);
            report.current_index = None;
            report.stage = "loop_delay".into();
            if policy.runs != 0 && report.runs_completed >= policy.runs {
                report.outcome = TaskOutcome::Success;
                report.stage = "completed".into();
                report.elapsed_ms = started.elapsed().as_millis() as u64;
                return report;
            }
            let delay_started = Instant::now();
            while delay_started.elapsed() < policy.delay {
                if host.is_cancelled() {
                    report.current_index = host.current_waypoint_index();
                    return Self::finish_cancelled(&mut report, started);
                }
                host.sleep_ms(50.min(policy.delay.as_millis().max(1) as u64));
            }
        }
    }

    fn wait_for_loading<H: PathingHost + ?Sized>(
        host: &mut H,
        timeout: Duration,
    ) -> Result<(), String> {
        let started = Instant::now();
        loop {
            if host.is_cancelled() {
                return Err("已取消".into());
            }
            if !host.detect_loading()? {
                return Ok(());
            }
            if started.elapsed() >= timeout {
                return Err("等待加载完成超时".into());
            }
            host.sleep_ms(200);
        }
    }

    fn finish_error(report: &mut TaskReport, started: Instant, error: String) -> TaskReport {
        if report.failure.is_none() {
            report.failure = Some(error);
        }
        report.outcome = if report.runs_completed > 0 {
            TaskOutcome::Partial
        } else {
            TaskOutcome::Failed
        };
        report.elapsed_ms = started.elapsed().as_millis() as u64;
        report.clone()
    }

    fn finish_cancelled(report: &mut TaskReport, started: Instant) -> TaskReport {
        report.outcome = TaskOutcome::Cancelled;
        report.elapsed_ms = started.elapsed().as_millis() as u64;
        report.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::tests::MockHost;

    fn task() -> PathingTaskFile {
        PathingTaskFile {
            positions: vec![crate::Waypoint {
                x: 20.0,
                y: 0.0,
                waypoint_type: "orientation".into(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn repeats_and_reports_success() {
        let mut host = MockHost::new((0.0, 0.0));
        let report = LoopRunner::run(
            &mut host,
            1,
            "loop",
            &task(),
            LoopPolicy {
                runs: 3,
                auto_pick: false,
                delay: Duration::ZERO,
                ..Default::default()
            },
        );
        assert_eq!(report.outcome, TaskOutcome::Success);
        assert_eq!(report.runs_completed, 3);
    }

    #[test]
    fn later_failure_is_partial() {
        let mut host = MockHost::new((0.0, 0.0));
        host.fail_on_focus = 2;
        let report = LoopRunner::run(
            &mut host,
            1,
            "loop",
            &task(),
            LoopPolicy {
                runs: 2,
                auto_pick: false,
                delay: Duration::ZERO,
                ..Default::default()
            },
        );
        assert_eq!(report.outcome, TaskOutcome::Partial);
        assert_eq!(report.runs_completed, 1);
        assert!(report.failure.is_some());
    }

    #[test]
    fn waits_for_loading_and_can_cancel_during_loading() {
        let mut host = MockHost::new((0.0, 0.0));
        host.loading = true;
        let report = LoopRunner::run(
            &mut host,
            1,
            "loop",
            &task(),
            LoopPolicy {
                wait_for_loading: true,
                auto_pick: false,
                ..Default::default()
            },
        );
        assert_eq!(report.outcome, TaskOutcome::Success);

        let mut host = MockHost::new((0.0, 0.0));
        host.loading = true;
        host.cancelled = true;
        let report = LoopRunner::run(
            &mut host,
            1,
            "loop",
            &task(),
            LoopPolicy {
                wait_for_loading: true,
                ..Default::default()
            },
        );
        assert_eq!(report.outcome, TaskOutcome::Cancelled);
        assert!(report.failure.is_none());
    }

    #[test]
    fn cancellation_before_start_has_no_input() {
        let mut host = MockHost::new((0.0, 0.0));
        host.cancelled = true;
        let report = LoopRunner::run(&mut host, 1, "loop", &task(), LoopPolicy::default());
        assert_eq!(report.outcome, TaskOutcome::Cancelled);
        assert!(host.events.is_empty());
    }
}
