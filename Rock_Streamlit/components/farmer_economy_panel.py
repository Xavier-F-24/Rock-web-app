import streamlit as st


def render_farmer_inventory(world, farm_id):
    farm = world.farm(farm_id)
    rows = [{"id": rock.id, "name": rock.name.full_name, "sex": rock.sex.value, "generation": rock.generation, "status": rock.status.value, "value": rock.value, "sell_value": rock.sell_value, "reserved": rock.id in world.reserved_rock_ids} for rock in sorted(farm.rocks.values(), key=lambda row: row.id)]
    st.dataframe(rows, width="stretch", hide_index=True)
