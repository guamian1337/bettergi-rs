use std::path::Path;

use crate::{PathingError, PathingInfo, PathingTaskFile, Waypoint};

#[derive(Debug, Clone)]
pub struct PathRecorder {
    task: PathingTaskFile,
}

impl PathRecorder {
    pub fn start(
        map_name: impl Into<String>,
        matching_method: impl Into<String>,
        bgi_version: impl Into<String>,
        x: f64,
        y: f64,
    ) -> Result<Self, PathingError> {
        validate_position(x, y)?;
        let mut task = PathingTaskFile {
            info: PathingInfo {
                name: "未命名路线".into(),
                bgi_version: Some(bgi_version.into()),
                task_type: "Collect".into(),
                map_name: map_name.into(),
                map_match_method: Some(matching_method.into()),
                ..PathingInfo::default()
            },
            ..PathingTaskFile::default()
        };
        task.positions.push(Waypoint {
            x,
            y,
            waypoint_type: "teleport".into(),
            ..Waypoint::default()
        });
        Ok(Self { task })
    }

    pub fn add_waypoint(
        &mut self,
        waypoint_type: impl Into<String>,
        x: f64,
        y: f64,
    ) -> Result<(), PathingError> {
        validate_position(x, y)?;
        let waypoint_type = waypoint_type.into();
        self.task.positions.push(Waypoint {
            x,
            y,
            waypoint_type: if waypoint_type.trim().is_empty() {
                "path".into()
            } else {
                waypoint_type
            },
            ..Waypoint::default()
        });
        Ok(())
    }

    pub fn task(&self) -> &PathingTaskFile {
        &self.task
    }

    pub fn into_task(self) -> PathingTaskFile {
        self.task
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PathingError> {
        self.task.save_to_file(path)
    }
}

fn validate_position(x: f64, y: f64) -> Result<(), PathingError> {
    if !x.is_finite() || !y.is_finite() {
        return Err(PathingError::InvalidRecordedPosition { x, y });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_initial_teleport_and_round_trips() {
        let mut recorder = PathRecorder::start("Teyvat", "SIFT", "1.2.3", 10.0, 20.0).unwrap();
        recorder.add_waypoint("target", 11.0, 21.0).unwrap();
        recorder.add_waypoint("", 12.0, 22.0).unwrap();
        let task = recorder.task();
        assert_eq!(task.positions.len(), 3);
        assert_eq!(task.positions[0].waypoint_type, "teleport");
        assert_eq!(task.positions[1].waypoint_type, "target");
        assert_eq!(task.positions[2].waypoint_type, "path");
        assert_eq!(task.info.bgi_version.as_deref(), Some("1.2.3"));

        let path = std::env::temp_dir().join(format!("bgi-record-{}.json", std::process::id()));
        recorder.save(&path).unwrap();
        let (_, restored) = crate::load_path_file_for_version(&path, "1.2.3").unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(restored.positions.len(), 3);
        assert_eq!(restored.positions[1].x, 11.0);
    }

    #[test]
    fn rejects_non_finite_recorded_positions() {
        assert!(matches!(
            PathRecorder::start("Teyvat", "SIFT", "1.0.0", f64::NAN, 0.0),
            Err(PathingError::InvalidRecordedPosition { .. })
        ));
    }
}
