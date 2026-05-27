import { useEffect, useState } from 'react';

type Props = {
  src: string;
  alt: string;
  className: string;
  fallbackText: string;
  logContext?: Record<string, unknown>;
};

export function ImageWithFallback({ src, alt, className, fallbackText, logContext }: Props) {
  const [failed, setFailed] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    setFailed(false);
    setRetryKey(0);
  }, [src]);

  if (failed) {
    return (
      <div className={`${className} image-fallback`} title={src}>
        <span>{fallbackText}</span>
        <button
          className="image-fallback__retry"
          type="button"
          onClick={() => {
            setRetryKey((current) => current + 1);
            setFailed(false);
          }}
        >
          Tải lại ảnh
        </button>
      </div>
    );
  }

  const displaySrc = src.startsWith('blob:') || src.startsWith('data:')
    ? src
    : `${src}${src.includes('?') ? '&' : '?'}retry=${retryKey}`;

  return (
    <img
      key={`${src}-${retryKey}`}
      className={className}
      src={displaySrc}
      alt={alt}
      onError={() => {
        console.error('image render failed', { src: displaySrc, alt, ...logContext });
        setFailed(true);
      }}
    />
  );
}
