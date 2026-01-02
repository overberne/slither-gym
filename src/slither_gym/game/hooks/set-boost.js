var boostRequestedAt = 9999999999999;
var maxBoostWaitTimeMs = 100;
const MinSpeed = 5.9; // Or 5.77777 sometimes
const MaxSpeed = 14.0;
/**
 * Enables or disables boost by dispatching Space key events.
 *
 * @param {boolean} enabled - true to enable boost, false to disable
 */
window.__setBoost = (enabled) => {
  if (slither === null) return;

  // If boost is requested but not granted, release the spacebar
  // and try again next setBoost
  if (
    slither.md &&
    slither.wmd &&
    enabled &&
    kd_u &&
    slither.sp <= MinSpeed &&
    Date.now() - boostRequestedAt > maxBoostWaitTimeMs
  ) {
    window.__setBoost(false);
    return;
  }

  // No duplicate events.
  if (enabled === slither.wmd) {
    return;
  }

  const eventType = enabled ? "keydown" : "keyup";
  const event = new KeyboardEvent(eventType, {
    key: " ",
    code: "Space",
    keyCode: 32,
    which: 32,
    bubbles: true,
    cancelable: true,
  });

  if (enabled) {
    document.onkeydown(event);
    boostRequestedAt = Date.now();
  } else {
    document.onkeyup(event);
  }
};
