# Bevy USD Composing

## Environment variables

These are optional runtime debug settings. All are off by default.

- `OPENUSD_DEBUG_STAGE`: Logs stage and spawn summaries to stdout and appends loader timings to `usd_loader.log`.
- `OPENUSD_DEBUG_DRAW`: Spawns scene bounds debug boxes.
- `OPENUSD_DEBUG_ORIGIN`: Spawns a small origin marker at (0,0,0).
- `OPENUSD_DEBUG_NOCULL`: Adds `NoFrustumCulling` on per-entity instances (legacy path). Instanced batches already disable culling.
- `OPENUSD_DEBUG_SKIP_INSTANCES`: Skips spawning USD instances (debug markers can still spawn).
- `OPENUSD_DEBUG_ENTITY_INSTANCES`: Forces the legacy per-instance entity spawn path (useful for per-instance components).
- `OPENUSD_DEBUG_INSTANCE_LIMIT`: Caps how many instances are spawned (applies to both spawn paths).
- `OPENUSD_DEBUG_INSTANCE_STRIDE`: Spawns every Nth instance (only applies when set to a value > 1).
