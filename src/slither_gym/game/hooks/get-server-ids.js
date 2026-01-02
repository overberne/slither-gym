/**
 * Gets the server ids sorted by ping in ascending order
 *
 * @returns {int[]} Server ids in ascending order of ping
 */
window.__getServerIds = () => {
  // From game.js showServers()
  sos.sort(function (a, b) {
    return parseFloat(a.ptm) - parseFloat(b.ptm);
  });
  for (var i = sos.length - 1; i >= 0; i--) {
    var sid = sos[i].sid;
    for (var j = sos.length - 1; j > i; j--) {
      var sid2 = sos[j].sid;
      if (sid == sid2) {
        if (sos[j].ptm > sos[i].ptm) sos.splice(j, 1);
        else sos.splice(i, 1);
        j = -1;
        break;
      }
    }
  }

  return sos.map((s) => s.sid);
};
