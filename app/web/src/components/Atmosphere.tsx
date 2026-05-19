/**
 * Background atmosphere: radial color washes (cyan + peach), conic aurora,
 * SVG grain, vignette. Sits at z-index 0; everything else stacks on top.
 *
 * Pure CSS for the gradients (see App.css `.atmosphere*` rules). The SVG
 * filter rides inline because it's the only way to make `feTurbulence`
 * portable.
 */
export function Atmosphere() {
  return (
    <div className="atmosphere" aria-hidden>
      <div className="atmosphere__radials" />
      <div className="atmosphere__aurora" />
      <svg className="atmosphere__noise" preserveAspectRatio="none">
        <filter id="parakeet-grain">
          <feTurbulence baseFrequency="0.9" numOctaves="2" />
        </filter>
        <rect width="100%" height="100%" filter="url(#parakeet-grain)" />
      </svg>
      <div className="atmosphere__vignette" />
    </div>
  )
}
