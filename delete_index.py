from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_eajwY_RTYhhEWWV1z7LbXSrFeLScmvTcz3MTNnpeCe8Lqcd6oGSV1N9GGAQQbVPePri3u")

index_name = "jarvis"

if index_name in [i["name"] for i in pc.list_indexes()]:
    pc.delete_index(index_name)
    print(f"🗑️ Deleted old index '{index_name}' successfully!")
else:
    print("Index not found — nothing to delete.")
