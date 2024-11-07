from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec

class PineconeVectordatabase():
    def __init__(self,api_key, dimension) -> None:
        # Initialize connection to Pinecone
        self.pc = PineconeGRPC(api_key=api_key)
        self.dimension = dimension

    def create_index(self,index_name):
        if not self.pc.has_index(index_name):
            self.pc.create_index(
                            index_name,
                            dimension=self.dimension,
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                        )
        pinecone_index = self.pc.Index(index_name)
        return pinecone_index
    
    def get_index(self,index_name):
        if self.pc.has_index(index_name):
            return self.pc.Index(index_name)
        return None
    
    def delete_index(self,index_name):
        self.pc.delete_index(index_name)

   
    
