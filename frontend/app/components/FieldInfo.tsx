'use client';

import { useId } from 'react';

type Props = {
  text: string;
  id?: string;
};

export default function FieldInfo({ text, id }: Props) {
  const autoId = useId();
  const tooltipId = id ?? `field-info-${autoId.replace(/[:]/g, '-')}`;

  return (
    <span className="fieldInfoWrap">
      <span
        className="fieldInfo"
        tabIndex={0}
        aria-label={text}
        aria-describedby={tooltipId}
        role="img"
      >
        i
      </span>
      <span id={tooltipId} className="fieldInfo__tooltip" role="note">
        {text}
      </span>
    </span>
  );
}
