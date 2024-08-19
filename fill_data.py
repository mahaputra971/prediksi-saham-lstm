from app.sql import get_issuer 

data_kode, data_nama = get_issuer()


for i in range(len(data_kode)):
    print(f"Kode: {data_kode[i]}, Nama: {data_nama[i]}")

print(f"panjang kode : {len(data_kode)}, panjang Nama: {len(data_nama)}")