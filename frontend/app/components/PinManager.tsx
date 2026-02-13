'use client';

import { useMemo } from 'react';

import CollapsibleCard from './CollapsibleCard';
import type { PinDisplayNode, PinSelectionId } from '../lib/types';

type Props = {
  nodes: PinDisplayNode[];
  selectedPinId: PinSelectionId | null;
  disabled: boolean;
  hasStop: boolean;
  canAddStop: boolean;
  oneStopHint?: string | null;
  onSelectPin: (id: PinSelectionId) => void;
  onRenameStop: (name: string) => void;
  onAddStop: () => void;
  onDeleteStop: () => void;
  onSwapPins: () => void;
  onClearPins: () => void;
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
  oneStopHint = null,
  onSelectPin,
  onRenameStop,
  onAddStop,
  onDeleteStop,
  onSwapPins,
  onClearPins,
}: Props) {
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
          disabled={disabled}
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
              aria-label={`${label} Name`}
            />
          ) : (
            <div className="pinManager__lockedTag" aria-label={`${label} Name Is Fixed`}>
              <span className="pinManager__lockedDot" aria-hidden="true">•</span>
              <span>Locked</span>
            </div>
          )}
        </div>
      </li>
    );
  }

  return (
    <CollapsibleCard title="Pins & Stops" dataTutorialId="pins.section">
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
                  disabled={disabled}
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
          disabled={disabled || !canAddStop}
          onClick={onAddStop}
          title={!canAddStop ? 'Set Both Start And End To Add A Stop.' : 'Add Midpoint Stop'}
          data-tutorial-action="pins.add_stop"
        >
          {hasStop ? 'Replace Stop' : 'Add Stop'}
        </button>
        <button
          type="button"
          className="secondary"
          disabled={disabled || !hasStop}
          onClick={onDeleteStop}
          title={!hasStop ? 'No Stop Exists To Delete.' : 'Delete Stop'}
          data-tutorial-action="pins.delete_stop"
        >
          Delete Stop
        </button>
        <button type="button" className="secondary" disabled={disabled} onClick={onSwapPins}>
          Swap Start/End
        </button>
        <button type="button" className="secondary" disabled={disabled} onClick={onClearPins}>
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
      {oneStopHint ? <div className="pinManager__hintError">{oneStopHint}</div> : null}
    </CollapsibleCard>
  );
}
