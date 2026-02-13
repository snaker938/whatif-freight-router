export type TutorialActionId = string;
export type TutorialLockScope = 'map_only' | 'sidebar_section_only' | 'free';

export type TutorialMapGuideSequencePoint = {
  pin: 'origin' | 'destination';
  city: 'newcastle' | 'london';
  lat: number;
  lon: number;
  radiusKm: number;
  zoom: number;
};

export type TutorialMapGuide = {
  mode: 'pin_sequence';
  sequence: TutorialMapGuideSequencePoint[];
};
export type TutorialPrefillId =
  | 'clear_map'
  | 'canonical_map'
  | 'canonical_setup'
  | 'canonical_advanced'
  | 'canonical_preferences'
  | 'canonical_departure'
  | 'canonical_duty'
  | 'canonical_oracle'
  | 'canonical_experiment';

export type TutorialRequirement = {
  actionId: TutorialActionId;
  label: string;
  kind?: 'ui' | 'manual';
  details?: string;
};

export type TutorialOptionalDecision = {
  id: string;
  label: string;
  actionIds: TutorialActionId[];
  defaultLabel: string;
};

export type TutorialStep = {
  id: string;
  chapterId: string;
  title: string;
  what: string;
  impact: string;
  targetIds: string[];
  required: TutorialRequirement[];
  optional?: TutorialOptionalDecision;
  prefillId?: TutorialPrefillId;
  allowMissingTarget?: boolean;
  lockScope?: TutorialLockScope;
  activeSectionId?: string;
  allowedActions?: string[];
  mapGuide?: TutorialMapGuide;
};

export type TutorialChapter = {
  id: string;
  title: string;
  description: string;
};

export type TutorialProgress = {
  version: 3;
  stepIndex: number;
  stepActionsById: Record<string, TutorialActionId[]>;
  optionalDecisionsByStep: Record<string, string[]>;
  updatedAt: string;
};

export type TutorialTargetRect = {
  top: number;
  left: number;
  width: number;
  height: number;
};
