import streamlit as st
row3 = st.columns(7, vertical_alignment="top", gap="small")
a3,b3,c3,d3,e3,f3,g3 =row3
st.html(
    """
    <div style='text-align: center;'>
        <h1>AI SECURITY CCTV SYSTEM</h1>
""",)

st.image("added/4.png")

st.html(
    """
    <div style='text-align: left;'>
        <p style='text-align: left; font-size: 15px; font-weight: lighter;'> Hai!<br>
            Kami menawarkan solusi inovatif untuk keamanan melalui sistem deteksi wajah dan people counting yang dapat diintegrasikan dengan CCTV. Dengan teknologi terkini, kami siap membantu Anda meningkatkan keamanan dan memantau aktivitas dengan lebih efektif. Terima kasih telah mengunjungi kami!

""",)


st.write("")

# Fitur Utama dengan gambar
st.html(
    """
    <div style='text-align: center;'>
        <h1 style ='font-size: 25px; font-weight: bolds; color: ##ff8c00;' >Fitur utama</h1>       
    """,)
st.write("")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("✅ **Face Recognizer**")
    st.image("added/31.png")  # Ganti URL gambar
    st.write("Sistem kami mampu mengenali wajah secara instan, bahkan di keramaian, dengan akurasi tinggi.")
    
with col2:
    
    st.write("✅ **People Counting**")
    st.image("added/2.png")  # Ganti URL gambar
    st.write("Menghitung jumlah orang di area publik untuk meningkatkan keamanan.")
    
with col3:
    st.write("✅ **Integrasi dengan CCTV**")
    st.image("added/3.png") 
    st.write("Kompatibel dengan sistem keamanan yang ada tanpa perangkat keras tambahan.")
st.write("")

st.html(
    """
    <div style='text-align: center;'>
        <h1 style ='font-size: 25px; font-weight: bolds; color: ##ff8c00;' >Studi Kasus</h1>       
    """,)
col4, col5, col6 = st.columns(3)

with col4:
    st.image("added/bandara.png", caption="Bandara")  # Ganti URL gambar
    st.write("**Bandara**: Meningkatkan pengawasan dengan face recognition untuk deteksi penumpang berisiko.")

with col5:
    st.image("added/mall.png", caption="Mall")  # Ganti URL gambar
    st.write("**Mall**: Penghitungan pengunjung otomatis untuk manajemen keramaian yang lebih baik.")

with col6:
    st.image("added/kantor.png", caption="Kantor")  # Ganti URL gambar
    st.write("**Kantor**: Implementasi akses kontrol menggunakan teknologi pengenalan wajah untuk meningkatkan keamanan.")

st.html(
    """
    <div style='text-align: center;'>
        <h1 style ='font-size: 20px; font-weight: bolds; color: ##ff8c00;' >Mulai Face Recognizer Sekarang!</h1>
            
    """,)
with st.container(height=380):
   
    row2 = st.columns(2, vertical_alignment="center", gap="small")
    a2, b2 =row2
    a2.image("added/31.png")
    b2.page_link("page/ambildataset.py", label="Input Dataset",icon=":material/add_circle:")
    b2.caption("Mulailah dengan mengumpulkan dan menyiapkan dataset wajah untuk memulai proses training. Pastikan kualitas dan variasi dataset optimal untuk hasil yang lebih akurat.")
    b2.write("")

    b2.page_link("page/faceRecognize.py", label="Deployment", icon=":material/photo_camera_front:")
    b2.caption("Aktifkan dan jalankan face recognizer yang telah dilatih untuk identifikasi wajah secara real-time.")



# FAQ Section
st.subheader("FAQ")
faq = {
    "Apakah sistem ini kompatibel dengan kamera keamanan yang sudah ada?": 
    "contoh jawaban..................",
    "Seberapa cepat sistem ini bisa mengenali wajah?": 
    "contoh jawaban..................",
    "Bagaimana dengan privasi data?": 
    "contoh jawaban.................."
}

for question, answer in faq.items():
    with st.expander(question):
        st.write(answer)

st.sidebar.subheader("Hubungi Kami")
st.sidebar.write("Ingin tahu lebih banyak tentang bagaimana AI CCTV SYSTEM dapat membantu Anda?")
st.sidebar.write("**Email**: support@[domainanda].com")
st.sidebar.write("**Telepon**: +62 [nomor telepon]")
st.sidebar.write("**Alamat**: Jl. [Alamat kantor]")
st.write("")
st.write("")
st.write("📌 Created by RNJ MBKM student of Brawijaya University")








