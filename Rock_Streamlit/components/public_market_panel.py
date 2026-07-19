import streamlit as st


def render_public_market(world):
    listings = []
    for listing in world.listings.values():
        listings.append({"listing": listing.listing_id, "seller": listing.seller_farm_id, "rock": listing.rock_id, "ask": listing.asking_price, "appraised": listing.appraised_value, "status": listing.status.value, "bids": len([bid for bid in listing.bids.values() if bid.active])})
    st.dataframe(listings, width="stretch", hide_index=True) if listings else st.info("No public listings yet.")
