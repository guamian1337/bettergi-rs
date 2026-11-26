use std::collections::VecDeque;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SubmitError {
    #[error("上一帧仍在执行")]
    Overlapping,
    #[error("调度队列已满")]
    QueueFull,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    pub submitted: u64,
    pub completed: u64,
    pub deadline_hits: u64,
    pub deadline_misses: u64,
    pub dropped: u64,
    pub queue_delay_total_us: u128,
    pub queue_delay_max_us: u128,
    sampled: u64,
}

impl SchedulerStats {
    pub fn deadline_hit_rate(&self) -> f32 {
        let total = self.deadline_hits + self.deadline_misses;
        if total == 0 {
            1.0
        } else {
            self.deadline_hits as f32 / total as f32
        }
    }

    pub fn average_queue_delay(&self) -> Duration {
        if self.sampled == 0 {
            Duration::ZERO
        } else {
            Duration::from_micros((self.queue_delay_total_us / self.sampled as u128) as u64)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameTicket {
    pub frame_id: u64,
    pub queue_delay: Duration,
    pub deadline_missed: bool,
}

#[derive(Debug, Clone)]
struct PendingFrame {
    frame_id: u64,
    deadline: Instant,
    enqueued_at: Instant,
}

#[derive(Debug)]
pub struct FrameScheduler {
    interval: Duration,
    capacity: usize,
    queue: VecDeque<PendingFrame>,
    running: bool,
    stats: SchedulerStats,
}

impl FrameScheduler {
    pub fn new(interval: Duration, capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            interval,
            capacity,
            queue: VecDeque::with_capacity(capacity),
            running: false,
            stats: SchedulerStats::default(),
        }
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }

    pub fn submit(
        &mut self,
        frame_id: u64,
        deadline: Instant,
        enqueued_at: Instant,
    ) -> Result<(), SubmitError> {
        if self.running {
            self.stats.dropped += 1;
            return Err(SubmitError::Overlapping);
        }
        if self.queue.len() >= self.capacity {
            self.stats.dropped += 1;
            return Err(SubmitError::QueueFull);
        }
        let pending = PendingFrame {
            frame_id,
            deadline,
            enqueued_at,
        };
        let position = self
            .queue
            .iter()
            .position(|frame| frame.deadline > pending.deadline)
            .unwrap_or(self.queue.len());
        self.queue.insert(position, pending);
        self.stats.submitted += 1;
        Ok(())
    }

    pub fn poll(&mut self, now: Instant) -> Option<FrameTicket> {
        if self.running {
            return None;
        }
        let pending = self.queue.front()?;
        if now < pending.deadline {
            return None;
        }
        let pending = self.queue.pop_front().unwrap();
        let queue_delay = now.saturating_duration_since(pending.enqueued_at);
        let deadline_missed = now > pending.deadline;
        if deadline_missed {
            self.stats.deadline_misses += 1;
        } else {
            self.stats.deadline_hits += 1;
        }
        self.stats.queue_delay_total_us += queue_delay.as_micros();
        self.stats.sampled += 1;
        self.stats.queue_delay_max_us = self.stats.queue_delay_max_us.max(queue_delay.as_micros());
        self.running = true;
        Some(FrameTicket {
            frame_id: pending.frame_id,
            queue_delay,
            deadline_missed,
        })
    }

    pub fn next_deadline(&self) -> Option<Instant> {
        self.queue.front().map(|frame| frame.deadline)
    }

    pub fn complete(&mut self) {
        self.running = false;
        self.stats.completed += 1;
    }

    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    Capture,
    Recognition,
    Decision,
    Input,
}

#[derive(Debug, Clone, Default)]
pub struct StageStats {
    pub samples: u64,
    pub total: Duration,
    pub max: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    stages: [StageStats; 4],
}

impl PipelineMetrics {
    pub fn observe(&mut self, stage: PipelineStage, elapsed: Duration) {
        let index = match stage {
            PipelineStage::Capture => 0,
            PipelineStage::Recognition => 1,
            PipelineStage::Decision => 2,
            PipelineStage::Input => 3,
        };
        let stats = &mut self.stages[index];
        stats.samples += 1;
        stats.total += elapsed;
        stats.max = stats.max.max(elapsed);
    }

    pub fn stage(&self, stage: PipelineStage) -> &StageStats {
        let index = match stage {
            PipelineStage::Capture => 0,
            PipelineStage::Recognition => 1,
            PipelineStage::Decision => 2,
            PipelineStage::Input => 3,
        };
        &self.stages[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_orders_deadlines_and_drops_overlapping_frames() {
        let now = Instant::now();
        let mut scheduler = FrameScheduler::new(Duration::from_millis(50), 2);
        scheduler
            .submit(2, now + Duration::from_millis(20), now)
            .unwrap();
        scheduler
            .submit(1, now + Duration::from_millis(10), now)
            .unwrap();

        let first = scheduler.poll(now + Duration::from_millis(10)).unwrap();
        assert_eq!(first.frame_id, 1);
        assert!(!first.deadline_missed);
        assert_eq!(scheduler.submit(3, now, now), Err(SubmitError::Overlapping));
        assert!(scheduler.poll(now + Duration::from_millis(20)).is_none());

        scheduler.complete();
        let second = scheduler.poll(now + Duration::from_millis(21)).unwrap();
        assert_eq!(second.frame_id, 2);
        assert!(second.deadline_missed);
        assert_eq!(scheduler.stats().deadline_hit_rate(), 0.5);
    }

    #[test]
    fn scheduler_exposes_the_next_deferred_deadline() {
        let now = Instant::now();
        let mut scheduler = FrameScheduler::new(Duration::from_millis(50), 2);
        let first_deadline = now + Duration::from_millis(50);
        let second_deadline = now + Duration::from_millis(100);
        scheduler.submit(1, first_deadline, now).unwrap();
        scheduler.submit(2, second_deadline, now).unwrap();
        assert_eq!(scheduler.next_deadline(), Some(first_deadline));
        assert!(scheduler.poll(now + Duration::from_millis(49)).is_none());
        assert!(scheduler.poll(first_deadline).is_some());
        assert_eq!(scheduler.next_deadline(), Some(second_deadline));
    }

    #[test]
    fn scheduler_enforces_queue_bound() {
        let now = Instant::now();
        let mut scheduler = FrameScheduler::new(Duration::from_millis(50), 1);
        scheduler
            .submit(1, now + Duration::from_secs(1), now)
            .unwrap();
        assert_eq!(scheduler.submit(2, now, now), Err(SubmitError::QueueFull));
        assert_eq!(scheduler.stats().dropped, 1);
    }

    #[test]
    fn pipeline_metrics_record_each_stage() {
        let mut metrics = PipelineMetrics::default();
        metrics.observe(PipelineStage::Capture, Duration::from_millis(4));
        metrics.observe(PipelineStage::Capture, Duration::from_millis(2));
        assert_eq!(metrics.stage(PipelineStage::Capture).samples, 2);
        assert_eq!(
            metrics.stage(PipelineStage::Capture).max,
            Duration::from_millis(4)
        );
    }
}
