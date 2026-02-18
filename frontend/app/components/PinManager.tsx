'use client';

import { useCallback, useEffect, useMemo } from 'react';

import CollapsibleCard from './CollapsibleCard';
import type { PinDisplayNode, PinSelectionId } from '../lib/types';

type Props = {
  nodes: PinDisplayNode[];
  selectedPinId: PinSelectionId | null;
  disabled: boolean;
  hasStop: boolean;
  canAddStop: boolean;
  tutorialRunning?: boolean;
  tutorialStepId?: string | null;
  tutorialBlockingActionId?: string | null;
  oneStopHint?: string | null;
  onSelectPin: (id: PinSelectionId) => void;
  onRenameStop: (name: string) => void;
  onAddStop: () => void;
  onDeleteStop: () => void;
  onSwapPins: () => void;
  onClearPins: () => void;
  sectionControl?: {
    isOpen?: boolean;
    lockToggle?: boolean;
    tutorialLocked?: boolean;
  };
};

function fmtCoord(value: number): string {
  return Number.isFinite(value) ? value.toFixed(5) : String(value);
}

export default function PinManager({
  nodes,
  selectedPinId,
  disabled,
  hasStop,
  canAddStop,
  tutorialRunning = false,
  tutorialStepId = null,
  tutorialBlockingActionId = null,
  oneStopHint = null,
  onSelectPin,
  onRenameStop,
  onAddStop,
  onDeleteStop,
  onSwapPins,
  onClearPins,
  sectionControl,
}: Props) {
  const logMidpointUi = useCallback(
    (_event: string, _payload?: Record<string, unknown>) => {},
    [],
  );
  const originNode = useMemo(() => nodes.find((node) => node.id === 'origin') ?? null, [nodes]);
  const stopNode = useMemo(() => nodes.find((node) => node.id === 'stop-1') ?? null, [nodes]);
  const destinationNode = useMemo(
    () => nodes.find((node) => node.id === 'destination') ?? null,
    [nodes],
  );
  const railNodes = useMemo(
    () => [originNode, stopNode, destinationNode].filter((node): node is PinDisplayNode => Boolean(node)),
    [originNode, stopNode, destinationNode],
  );
  const strictStepActionMode =
    tutorialRunning && tutorialStepId === 'map_stop_lifecycle';
  const isActionEnabled = (actionId: string) => {
    if (!strictStepActionMode) return true;
    return tutorialBlockingActionId === actionId;
  };
  const addStopDisabled = disabled || !canAddStop;
  const addStopDisabledReason = addStopDisabled
    ? disabled
      ? 'panel-disabled'
      : 'missing-start-or-end'
    : !isActionEnabled('pins.add_stop')
      ? 'blocked-by-tutorial-action'
      : null;
  const addStopActionEnabled = isActionEnabled('pins.add_stop');

  useEffect(() => {
    if (!(tutorialRunning && tutorialStepId === 'map_stop_lifecycle')) return;
    logMidpointUi('state', {
      disabled,
      canAddStop,
      hasStop,
      addStopDisabled,
      addStopDisabledReason,
      addStopActionEnabled,
      tutorialRunning,
      tutorialStepId,
      tutorialBlockingActionId,
      strictStepActionMode,
      selectedPinId,
      nodes: nodes.map((node) => ({
        id: node.id,
        kind: node.kind,
        lat: node.lat,
        lon: node.lon,
        label: node.label,
      })),
      sectionOpen: sectionControl?.isOpen ?? null,
      sectionLocked: sectionControl?.tutorialLocked ?? null,
      panelDisabled: disabled,
    });
  }, [
    addStopActionEnabled,
    addStopDisabled,
    addStopDisabledReason,
    canAddStop,
    disabled,
    hasStop,
    logMidpointUi,
    nodes,
    sectionControl?.isOpen,
    sectionControl?.tutorialLocked,
    selectedPinId,
    strictStepActionMode,
    tutorialBlockingActionId,
    tutorialRunning,
    tutorialStepId,
  ]);

  const handleAddStopClick = useCallback(() => {
    logMidpointUi('add-stop-click', {
      addStopDisabled,
      addStopDisabledReason,
      addStopActionEnabled,
      tutorialRunning,
      tutorialStepId,
      tutorialBlockingActionId,
      canAddStop,
      hasStop,
      disabled,
    });
    onAddStop();
  }, [
    addStopActionEnabled,
    addStopDisabled,
    addStopDisabledReason,
    canAddStop,
    disabled,
    hasStop,
    logMidpointUi,
    onAddStop,
    tutorialBlockingActionId,
    tutorialRunning,
    tutorialStepId,
  ]);

  function renderNodeRow(
    node: PinDisplayNode,
    label: string,
    options: {
      editable: boolean;
      onRename?: (name: string) => void;
      description: string;
    },
  ) {
    const isSelected = selectedPinId === node.id;
    return (
      <li key={node.id} className={`pinManager__row ${isSelected ? 'isSelected' : ''}`}>
        <button
          type="button"
          className="pinManager__nodeBtn"
          onClick={() => onSelectPin(node.id)}
          disabled={disabled || !isActionEnabled('pins.sidebar_select')}
          aria-pressed={isSelected}
          aria-label={`${isSelected ? 'Deselect' : 'Select'} ${label} Pin`}
          data-tutorial-action="pins.sidebar_select"
        >
          <span className={`pinManager__dot pinManager__dot--${node.kind}`} style={{ background: node.color }} />
          <span className="pinManager__nodeText">
            <span className="pinManager__nodeTitle">{label}</span>
            <span className="pinManager__nodeCoords">
              {fmtCoord(node.lat)}, {fmtCoord(node.lon)}
            </span>
            <span className="pinManager__nodeHint">{options.description}</span>
          </span>
        </button>

        <div className="pinManager__inlineActions">
          {options.editable ? (
            <input
              className="input pinManager__nameInput"
              value={node.label}
              onChange={(event) => options.onRename?.(event.target.value)}
              placeholder={node.label || 'Stop #1'}
              disabled={disabled}
              readOnly={!isActionEnabled('pins.rename_stop')}
              aria-label={`${label} Name`}
              data-tutorial-action="pins.rename_stop"
            />
          ) : (
            <div className="pinManager__fixedName" aria-label={`${label} Name Is Fixed`}>
              <span className="pinManager__fixedNameDot" aria-hidden="true">●</span>
              <span>{label} Name Is Fixed</span>
            </div>
          )}
        </div>
      </li>
    );
  }

  return (
    <CollapsibleCard
      title="Pins & Stops"
      dataTutorialId="pins.section"
      isOpen={sectionControl?.isOpen}
      lockToggle={sectionControl?.lockToggle}
      tutorialLocked={sectionControl?.tutorialLocked}
    >
      <div className="pinManager__rail" role="list" aria-label="Pin Route Order">
        {railNodes.length === 0 ? (
          <span className="pinManager__railEmpty">Add Start And End Pins To Build A Route.</span>
        ) : (
          railNodes.map((node, idx) => {
            const isSelected = selectedPinId === node.id;
            return (
              <div key={node.id} className="pinManager__railItem" role="listitem">
                <button
                  type="button"
                  className={`pinManager__railNode ${isSelected ? 'isSelected' : ''}`}
                  style={{ color: node.color }}
                  disabled={disabled || !isActionEnabled('pins.sidebar_select')}
                  onClick={() => onSelectPin(node.id)}
                  aria-pressed={isSelected}
                  aria-label={`${isSelected ? 'Deselect' : 'Select'} ${node.label}`}
                >
                  <span className="pinManager__railNodeText">{node.label}</span>
                </button>
                {idx < railNodes.length - 1 ? (
                  <span className="pinManager__arrow" aria-hidden="true">→</span>
                ) : null}
              </div>
            );
          })
        )}
      </div>

      <ul className="pinManager__list">
        {originNode
          ? renderNodeRow(originNode, 'Start', {
              editable: false,
              description: 'Route Start Point.',
            })
          : null}
        {stopNode
          ? renderNodeRow(stopNode, stopNode.label || 'Stop #1', {
              editable: true,
              onRename: onRenameStop,
              description: 'Optional Intermediate Stop. Name Is Editable.',
            })
          : null}
        {destinationNode
          ? renderNodeRow(destinationNode, 'End', {
              editable: false,
              description: 'Route End Point.',
            })
          : null}
      </ul>

      <div className="pinManager__actions">
        <button
          type="button"
          className="secondary"
          disabled={addStopDisabled || !addStopActionEnabled}
          onClick={handleAddStopClick}
          title={!canAddStop ? 'Set Both Start And End To Add A Stop.' : 'Add Midpoint Stop'}
          data-tutorial-action="pins.add_stop"
        >
          {hasStop ? 'Replace Stop' : 'Add Stop'}
        </button>
        <button
          type="button"
          className="secondary"
          disabled={disabled || !hasStop || !isActionEnabled('pins.delete_stop')}
          onClick={onDeleteStop}
          title={!hasStop ? 'No Stop Exists To Delete.' : 'Delete Stop'}
          data-tutorial-action="pins.delete_stop"
        >
          Delete Stop
        </button>
        <button
          type="button"
          className="secondary"
          disabled={disabled || !isActionEnabled('pins.swap_start_end')}
          onClick={onSwapPins}
          data-tutorial-action="pins.swap_start_end"
        >
          Swap Start/End
        </button>
        <button
          type="button"
          className="secondary"
          disabled={disabled || !isActionEnabled('pins.clear_pins')}
          onClick={onClearPins}
          data-tutorial-action="pins.clear_pins"
        >
          Clear Pins
        </button>
      </div>

      <div className="pinManager__reorder">
        <button
          type="button"
          className="secondary"
          disabled
          title="Unavailable In One-Stop Mode"
          aria-disabled="true"
        >
          Move Stop Up
        </button>
        <button
          type="button"
          className="secondary"
          disabled
          title="Unavailable In One-Stop Mode"
          aria-disabled="true"
        >
          Move Stop Down
        </button>
      </div>
      <div className="tiny">Reorder Is Disabled In One-Stop Mode. Enable When Multi-Stop Support Is Added.</div>
      <div className="tiny">Tip: Click Into Stop #1 Name To Rename, Then Drag The Stop Marker On The Map.</div>
      {strictStepActionMode ? (
        <div className="tiny">
          Tutorial tip: you can also use marker popups on Start/End to add a midpoint in normal map workflows.
        </div>
      ) : null}
      {oneStopHint ? <div className="pinManager__hintError">{oneStopHint}</div> : null}
    </CollapsibleCard>
  );
}
