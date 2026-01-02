/**
 * Sets the snake heading and controls boost.
 *
 * @param {number} angleRadians - Angle in radians.
 * @param {boolean} enableBoost - true to enable boost, false to disable
 */
window.__act = (angleRadians, enableBoost) => {
  window.__setHeading(angleRadians);
  window.__setBoost(enableBoost);
};
