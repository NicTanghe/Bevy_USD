# bevy_usd_lib

Bevy USD loader with labeled sub-assets and explicit spawn control.

## Usage

Add the plugin and load the root USD asset or a labeled scene:

```rust
use bevy::prelude::*;
use bevy_usd_lib::{Usd, UsdPlugin, UsdScene, UsdSceneBundle};

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let _usd: Handle<Usd> = asset_server.load("models/scene.usd");
    let scene: Handle<UsdScene> = asset_server.load("models/scene.usd#Scene0");
    commands.spawn(UsdSceneBundle::new(scene));
}
```

## Labels

- `Scene0` -> whole stage (pseudo-root).
- `Scene:/PrimPath` -> top-level prim scene.
- `Mesh:/PrimPath` -> mesh sub-asset.
- `Material:/PrimPath` -> material sub-asset.

## Spawn metadata

Each spawned mesh entity includes:

- `UsdPrimPath` (full prim path)
- `UsdPrimName` (prim name)

You can override materials after spawn by swapping `Handle<StandardMaterial>` values.

## Notes

The loader resolves USD file paths against:

1. The configured `UsdLoaderSettings.root_path` (if set)
2. The current working directory
3. `./assets` in the current working directory
