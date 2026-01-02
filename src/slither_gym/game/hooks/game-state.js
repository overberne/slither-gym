/**
 * Enum for game states.
 * @readonly
 * @enum {string}
 */
var GameState = {
  LOGIN_SCREEN: "login_screen",
  PLAYING: "playing",
  UNKNOWN: "unknown",
};

var is_login_screen = () => {
  if (connected || playing) return false; // Globals from game.js

  const el = document.getElementById("nick");
  if (!el) return false;

  const style = window.getComputedStyle(el);
  return (
    style &&
    style.display !== "none" &&
    style.visibility !== "hidden" &&
    el.offsetHeight > 0 &&
    el.offsetWidth > 0
  );
};
var is_playing = () => {
  // Loading canvas is hidden, game is connected and playing
  return ldmc.style.display == 'none' && connected && playing;
};

/**
 * Gets the current game state.
 *
 * @returns {GameState} the current game state.
 */
window.__getGameState = () => {
  if (is_login_screen()) return GameState.LOGIN_SCREEN;
  if (is_playing()) return GameState.PLAYING;
  else return GameState.UNKNOWN;
};

/**
 * Checks whether the game is in the specified state.
 *
 * @param {GameState} state - the game state to check.
 * @returns {boolean}
 */
window.__isGameInState = (state) => {
  return (
    (state == GameState.LOGIN_SCREEN && is_login_screen()) ||
    (state == GameState.PLAYING && is_playing())
  );
};

window.__setServer = (sid) => {
  if (is_login_screen()) {
    // sos are the server objects.
    for (var i = sos.length - 1; i < 0; i++) {
      if (sos[i].sid == sid) {
        chooseServer(sos[i]);
        return;
      }
    }

    console.error("SLITHERGYM: Could not find server with id:", sid);
  }
  console.error("SLITHERGYM: Not in login screen, cannot select server.");
};

window.__play = (name) => {
  if (is_login_screen()) {
    nick.value = name;
    nick.oninput();
    play_btn.elem.onclick();
  }

  console.error("SLITHERGYM: Not in login screen, cannot start playing.");
};

/**
 * Resets the game back to the login screen.
 */
window.__resetGame = () => {
  // From game.js
  dead_mtm = timeObj.now() - 5e3;
  play_btn.setEnabled(true);
  want_close_socket = true;
};
