/**
 * Dispatches a mousemove event at a point on the inscribed circle
 * of the viewport's client rectangle corresponding to the given angle.
 *
 * @param {number} angleRadians - Angle in radians.
 */
window.__setHeading = (angleRadians) => {
  const rect = document.documentElement.getBoundingClientRect();

  // Center of the client rectangle, used by slither.
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  // Radius of the inscribed circle
  const radius = Math.min(rect.width, rect.height) / 2;
  // Coordinates on the circle
  const x = centerX + radius * Math.cos(angleRadians);
  const y = centerY + radius * Math.sin(angleRadians);

  const event = new MouseEvent("mousemove", {
    clientX: x,
    clientY: y,
    bubbles: true,
    cancelable: true,
    view: window,
  });

  document.dispatchEvent(event);
};
