import streamlit as st

home = st.Page("page/home.py", title="Home", icon=":material/home:", default = True)
ambildataset = st.Page("page/ambildataset.py", title="Ambil Dataset", icon=":material/add_circle:")
facerecognizer = st.Page("page/faceRecognize.py", title="Deployment", icon=":material/photo_camera_front:")
peoplecounting = st.Page("page/pc1.py", title="People Counting", icon=":material/groups:")
CCTVfr = st.Page("page/CCTVfr.py", title="Face Recognition", icon=":material/photo_camera_front:")
pt = st.Page("page/testpa.py", title="People tracker", icon=":material/person_check:")
pa = {
    "Home": [home],
    "Face Recognation": [ambildataset, facerecognizer],
    "Integrasi CCTV": [CCTVfr, pt, peoplecounting]

}

logo1="added/4.png"
logo2="added/31.png"
st.logo(
    logo1, icon_image=logo2,
)
pg = st.navigation(pa)
pg.run()
