pub mod artifact_salvage;
pub mod artifact_salvage_preview;
pub mod auto_cook;
pub mod auto_eat;
pub mod auto_open_chest;
pub mod auto_pick;
pub mod auto_skip;
pub mod auto_wood;
pub mod character_attribute_recognition;
pub mod cooking_bar;
pub mod expedition_recognition;
pub mod fault;
pub mod fishing;
pub mod lease;
pub mod mining;
pub mod one_key_expedition;
pub mod reward_claim;
pub mod reward_result_recognition;
pub mod scheduler;
pub mod talent_recognition;
pub mod trigger;
pub mod weapon_recognition;

pub use artifact_salvage::{
    ArtifactCard, MAX_SALVAGE_ITEMS, SalvageConfig, SalvageError, SalvagePreview,
    authorize_final_confirmation, plan_salvage_preview,
};
pub use artifact_salvage_preview::{
    MAX_SALVAGE_CELLS_PER_PAGE, MAX_SALVAGE_INTERACTIONS, MAX_SALVAGE_NO_PROGRESS,
    MAX_SALVAGE_PREVIEW_PAGES, SalvageCandidate, SalvageCardObservation, SalvageCellRect,
    SalvagePosition, SalvagePreviewAction, SalvagePreviewConfig, SalvagePreviewError,
    SalvagePreviewFrame, SalvagePreviewReport, SalvagePreviewStateMachine,
    SalvagePreviewStopReason, SalvageScrollRequest, SalvageSkipStats,
};
pub use auto_cook::{
    AutoCookTaskConfig, CookingConfig, CookingConfigError, CookingEvent, CookingEvents,
    CookingFrame, CookingInputError, CookingState, CookingStateMachine,
};
pub use auto_eat::{AutoEatAction, AutoEatController, AutoEatObservation};
pub use auto_open_chest::{
    AUTO_OPEN_CHEST_COOLDOWN, AUTO_OPEN_CHEST_INTERACT_DELAY, AUTO_OPEN_CHEST_TICK,
    AUTO_OPEN_CHEST_TIMEOUT, AUTO_OPEN_CHEST_VERIFY_TIMEOUT, AutoOpenChestAction,
    AutoOpenChestConfig, AutoOpenChestController,
};
pub use auto_pick::{
    AUTO_PICK_MAX_OBSERVATION_AGE, AUTO_PICK_REQUIRED_STABLE_SAMPLES, AutoPickConfig,
    AutoPickDecision, AutoPickGateState, AutoPickMatcher, AutoPickMode, AutoPickPrompt,
    AutoPickPromptGate, AutoPickReason, AutoPickTextRoi, OcrCandidate, RoiCenter,
};
pub use auto_skip::{AutoSkipController, AutoSkipEvent};
pub use auto_wood::{
    AUTO_WOOD_AFTER_USE_WAIT, AUTO_WOOD_EMPTY_OCR_LIMIT, AUTO_WOOD_GAIN_TIMEOUT,
    AUTO_WOOD_LOGIN_TIMEOUT, AUTO_WOOD_TICK, AUTO_WOOD_TOTAL_TIMEOUT, AutoWoodAction,
    AutoWoodConfig, AutoWoodController, AutoWoodObservation, AutoWoodOcrSample, parse_wood_gain,
};
pub use character_attribute_recognition::{
    CHARACTER_ATTRIBUTE_MAX_OCR_ATTEMPTS, CHARACTER_ATTRIBUTE_REQUIRED_STABLE_SAMPLES,
    CharacterAttributeDetails, CharacterAttributePixelRoi, CharacterAttributeRecognitionStability,
    CharacterAttributeRecognitionState, character_attribute_roi, parse_character_level_pair,
};
pub use cooking_bar::{CookingBarConfig, CookingBarDetector, CookingBarEvent};
pub use expedition_recognition::{
    EXPEDITION_MAX_ATTEMPTS, EXPEDITION_REQUIRED_STABLE_SAMPLES, ExpeditionPageState,
    ExpeditionRecognitionStability, ExpeditionRecognitionState,
};
pub use fault::{Fault, FaultBoundary, FaultInjector};
pub use fishing::{
    FishingAction, FishingBar, FishingConfig, FishingController, FishingObservation, FishingRect,
};
pub use lease::{
    PauseController, PauseLease, TriggerLease, TriggerLeaseRegistry, TriggerLeaseSnapshot,
};
pub use mining::{
    MINING_MAX_ATTEMPTS, MINING_REQUIRED_STABLE_SAMPLES, MiningDetection, MiningObservation,
    MiningRecognitionStability, MiningRecognitionState, MiningRect, MiningTarget,
    filter_ore_detections, select_nearest_mining_target,
};
pub use one_key_expedition::{
    ONE_KEY_EXPEDITION_COLLECT_DELAY, ONE_KEY_EXPEDITION_RESPONSE_DELAY, ONE_KEY_EXPEDITION_TICK,
    ONE_KEY_EXPEDITION_TIMEOUT, ONE_KEY_EXPEDITION_VERIFY_TIMEOUT, OneKeyExpeditionAction,
    OneKeyExpeditionConfig, OneKeyExpeditionController, OneKeyExpeditionTarget,
};
pub use reward_claim::{
    MAX_CLICKS_PER_RUN, MAX_EMPTY_SCANS, RewardCandidate, RewardClaimAction, RewardClaimConfig,
    RewardClaimController, RewardClaimFinish, RewardClaimMode, RewardKind,
};
pub use reward_result_recognition::{
    REWARD_RESULT_MAX_ATTEMPTS, REWARD_RESULT_REQUIRED_STABLE_SAMPLES, RewardCardObservation,
    RewardCardRect, RewardIdentity, RewardPageMerge, RewardPageMergeError, RewardPageObservation,
    RewardResultRecognitionStability, RewardResultRecognitionState, merge_reward_pages,
    parse_reward_count, summarize_reward_cards,
};
pub use scheduler::{
    FrameScheduler, FrameTicket, PipelineMetrics, PipelineStage, SchedulerStats, SubmitError,
};
pub use talent_recognition::{
    TALENT_MAX_OCR_ATTEMPTS, TALENT_REQUIRED_STABLE_SAMPLES, TalentDetails, TalentPixelRoi,
    TalentRecognitionStability, TalentRecognitionState, TalentRois, TalentType,
    is_talent_detail_text, parse_talent_bonus, parse_talent_level, parse_talent_type, talent_rois,
};
pub use trigger::{
    Decision, DecisionAction, FrameContext, RuntimeError, Trigger, TriggerDispatcher,
    TriggerFailure, TriggerMode, TriggerSpec, UiCategory,
};
pub use weapon_recognition::{
    WEAPON_MAX_OCR_ATTEMPTS, WEAPON_REQUIRED_STABLE_SAMPLES, WeaponDetails, WeaponLevelPair,
    WeaponPixelRoi, WeaponRecognitionStability, WeaponRecognitionState, WeaponRois,
    parse_weapon_level_pair, weapon_rois,
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaskId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RunId(pub u64);

#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Generation(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowSignal {
    Retry,
    RetryNoCount,
    NormalEnd,
    Handled,
    Cancelled,
}
