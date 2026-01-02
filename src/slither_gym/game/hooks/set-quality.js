/**
 * Sets the game's graphical quality
 * @param {boolean} high - true if high quality is desired
 */
window.__setQuality = (high) => {
  if (high) {
    localStorage.qual = "1";
    grqi.src = "http://slither.io/s/highquality.png";
    want_quality = 1;
  } else {
    localStorage.qual = "0";
    grqi.src = "http://slither.io/s/lowquality.png";
    want_quality = 0;
  }
};
