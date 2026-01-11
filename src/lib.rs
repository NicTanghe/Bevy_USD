mod labels;
mod loader;
mod open_rs_loader;
mod spawn;
mod types;
mod usdish;

use bevy::asset::AssetApp;
use bevy::prelude::{App, Plugin, Update};

pub use labels::{material_label_for_path, mesh_label_for_path, scene_label_for_path, USD_SCENE0_LABEL};
pub use loader::{UsdLoader, UsdLoaderSettings};
pub use spawn::spawn_usd_scenes;
pub use types::{
    Usd, UsdInstance, UsdPrimName, UsdPrimPath, UsdScene, UsdSceneBundle, UsdSceneRoot,
    UsdSceneSpawned,
};

pub struct UsdPlugin;

impl Plugin for UsdPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Usd>()
            .init_asset::<UsdScene>()
            .register_asset_loader(UsdLoader)
            .add_systems(Update, spawn::spawn_usd_scenes);
    }
}
