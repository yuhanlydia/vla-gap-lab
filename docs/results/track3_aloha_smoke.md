# Track 3 Aloha trajectory smoke

Date: 2026-09-01

The official RoboTwin repository and one official XPolicyLab archive
(`blocks_ranking_rgb`, 152 Aloha episodes) were used to validate the complete
data path. A six-state smoke cache from two episodes contains:

- three decoded camera views per state (`240×320` RGB);
- the official-client-compatible 20D dual-arm EE6D proprio vector;
- five Florence encoder layers (`5×1024` per state);
- seven domain-conditioned action-transformer layers (`7×1024` per state);
- X-VLA actions (`30×20`).

All tensors are finite. This does **not** test cross-embodiment transport:
RoboTwin's downloadable pre-collected archive contains Aloha only. The current
`progress` label is normalized trajectory time and is explicitly a proxy, not
semantic task-phase ground truth. Franka/ARX paired replays or separately
generated demonstrations remain required before applying the Track 3 Gate.
