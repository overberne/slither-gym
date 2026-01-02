/**
 * Converts the pts (segments) of a slither into segments
 * with relative positions to the previous segment.
 *
 * This excludes the head as it would always be (0, 0)
 * @param {Array<{xx: number, yy: number}>} pts - the `pts` property from a slither object
 * @returns {Array<{dx: number, dy: number}>}
 */
function getRelativeSegments(pts) {
  // var segments = [{ dx: 0, dy: 0 }];
  const segments = [];
  if (!pts || pts.length < 2) return segments;

  for (var i = 1; i < pts.length; i++) {
    segments.push({
      dx: pts[i].xx - pts[i - 1].xx,
      dy: pts[i].yy - pts[i - 1].yy,
    });
  }

  return segments;
}

window.__pollObservation = function () {
  const sct = slither.sct + slither.rsc; // From game.js
  const out = {
    player_snake: {},
    enemy_snakes: [],
    food: [],
    score: Math.floor((fpsls[sct] + slither.fam / fmlts[sct] - 1) * 15 - 5),
    world_center: grd,
    world_radius: real_flux_grd,
    minimap: [],
    terminated: false,
  };

  // Death
  if (dead_mtm != -1 || !playing || !connected) {
    out.terminated = true;
    return out;
  } else if (!slither) {
    return out;
  }

  // Snakes
  for (var i = 0; i < slithers.length; i++) {
    const s = slithers[i];
    // iiv = is in view
    if (!s || !s.alive || !o.iiv) continue;

    const snake = {
      x: s.xx,
      y: s.yy,
      speed: s.sp || 0.0,
      heading: s.ang || 0.0,
      intended_heading: s.eang || 0.0,
      length: (s.pts?.length || 0.0).toFixed(1),
      boosting: s.md,
      segments: getRelativeSegments(s.pts),
    };

    if (s === slither) {
      out.player_snake = snake;
    } else {
      // Enemy snake position relative to player snake
      snake.xx -= out.player_snake.xx;
      snake.yy -= out.player_snake.yy;
      out.enemy_snakes.push(snake);
    }
  }

  // Food filtering bounds
  fpx1 = view_xx - (mww2 / gsc + 24);
  fpy1 = view_yy - (mhh2 / gsc + 24);
  fpx2 = view_xx + (mww2 / gsc + 24);
  fpy2 = view_yy + (mhh2 / gsc + 24);

  // Food
  for (var i = foods_c - 1; i >= 0; i--) {
    const fo = foods[i];
    if (fo.rx >= fpx1 && fo.ry >= fpy1 && fo.rx <= fpx2 && fo.ry <= fpy2) {
      out.food.push({ x: fo.xx, y: fo.yy, size: fo.sz });
    }
  }

  // Minimap
  const normalising_constant = (mmsz - 1) / 2; // Normalize to [0, 2]
  const centering_bias = 1; // Bias to [-1, 1]
  for (var i = mmdata.length - 1; i >= 0; i--) {
    if (mmdata[i] === 0) continue;

    // Normalize and center the coordinates around the map center.
    out.minimap.push({
      x: (i % mmsz) / normalising_constant - centering_bias,
      y: Math.floor(i / mmsz) / normalising_constant - centering_bias,
    });
  }
  return out;
};
