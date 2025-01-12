import yaml
from llama_index.core import StorageContext, VectorStoreIndex , Settings, PromptTemplate
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.evaluation import AnswerRelevancyEvaluator, ContextRelevancyEvaluator
from llama_index.readers.file.docs import PDFReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.llms.openai import OpenAI
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore
from db_utils import   create_connection_string, create_database, create_table


class VectorStore:

    def __init__(self):

        self.query_engine = None
        self.sentence_index = None
        self.vector_store = None
        self.storage_context = None
        self.sentence_window_engine = None

        self.initialize_table()

    def load_config(self, config_file="config.yml"):

        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def initialize_table(self):

        config = self.load_config('config.yml')

        create_database(config['db_name'])
        create_table(config['table_name'])


    def create_vector_store(self, conn_string,db_name, table_name):

        print('creating vector store ....')
        try:
            url = make_url(conn_string)
        except Exception as e:
            raise ValueError(f"Invalid connection string: {conn_string}") from e

            # Create the vector store
        try:
            self.vector_store = PGVectorStore.from_params(
                database=db_name,
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
                table_name=table_name,
                embed_dim=1536,

                hnsw_kwargs={
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 64,
                    "hnsw_ef_search": 40,
                    "hnsw_dist_method": "vector_cosine_ops",
                },
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create vector store for table {table_name}.") from e



    def create_store_index(self, documents, llm, embed_model, conn_string, db_name, table_name, sentence_window_size=3,):

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=sentence_window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        # sentence_context = ServiceContext.from_defaults(
        #     llm=llm,
        #     embed_model=embed_model,
        #     node_parser=node_parser,
        # )

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser



        self.create_vector_store(conn_string, db_name, table_name)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)


        self.sentence_index = VectorStoreIndex.from_documents(
            documents, storage_context=self.storage_context
        )
        self.storage_context.persist()


    def process_document(self, file_path):

        print('processing uploaded documents')

        reader = PDFReader()
        documents = reader.load_data(file_path)

        for document in documents:
            document.text = document.text.replace('\x00', '')

        if self.sentence_index is None:
            print('index do not exist. creating a new index')
            config = self.load_config('config.yml')
            conn_string = create_connection_string()
            db_name = config.get("db_name")
            table_name = config.get("table_name")
            embed_model = config.get("embed_model")
            sentence_window_size = config.get("sentence_window_size")
            llm_model = config.get("llm_model")

            system_prompt = """You are a helpful AI assistant. When the user answer a question provide clear, concise, 
            and accurate responses. If the question is not related to any specific document, do not user your existing knowledge to answer. Politely say the 
            question is not relevant to the provided document and be kind and ask if need any more assistance"""

            llm = OpenAI(model=llm_model, temperature=0.1, system_prompt=system_prompt)

            embed_llm = OpenAIEmbedding(model=embed_model)

            self.create_store_index(documents, llm, embed_llm, conn_string, db_name, table_name, sentence_window_size )

            print('created store index')

        else:

            print('index exists. Persisting new documents')
            for d in documents:
                self.sentence_index.insert(document=d, storage_context=self.storage_context)
            self.storage_context.persist()



    def create_query_engine(self,  similarity_top_k=6, rerank_top_n=2):

        cohere_rerank = CohereRerank(top_n=rerank_top_n)

        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

        self.query_engine = self.sentence_index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, cohere_rerank]
        )

    def evaluate_response(self,  query, response):

        context = [node.dict()['node']['text'] for node in response.source_nodes]

        context_relevancy = ContextRelevancyEvaluator(llm=OpenAI(temperature=0, model="gpt-4"))

        answer_relevancy = AnswerRelevancyEvaluator(
            llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
        )

        answer_score = answer_relevancy.evaluate_response(query, response).score
        context_score = context_relevancy.evaluate(query, response, context).score
        print('answer relevancy score ' + str(answer_score))
        print('context relevancy score' + str(context_score))
        return answer_score, context_score

    def provide_response(self, query, response):

        answer_relevancy, context_relevancy = self.evaluate_response(query, response)
        chat_template = """
        You are an AI assistant tasked with providing answers based on the given query, answer, and relevancy scores.

        Question: "{query}"

        Answer: "{response}"

        Answer Relevancy Score: {answer_relevancy}

        Context Relevancy Score: {context_relevancy}

        Instructions:
        - If **both** the Answer Relevancy Score and Context Relevancy Score indicate a strong match, respond **only** with the provided Answer. Do not include any additional information. If either one score is low follow below rules.
        - If the Answer Relevancy Score is low, do not attempt to answer the question. Instead, reply: "I'm sorry, but I couldn't confidently determine an answer to your question. Could you please rephrase or clarify your query? I'm here to help!"
        - If the Context Relevancy Score is low, explain that the question is outside the scope of the provided documents. Suggest asking another question related to the context of the documents.
        - Do not explain the reasoning behind your response or mention the relevancy scores in your reply.
        
        Your responses should strictly follow the above rules.
        """.format(query=query, response=response, answer_relevancy=answer_relevancy,
                   context_relevancy=context_relevancy)

        llm = OpenAI(model="gpt-4", temperature=0.1)
        # Call the LLM
        output = llm.predict(PromptTemplate(chat_template))
        print('final output ' + str(output))
        return output

    def process_msg(self, msg):

        if self.query_engine is None:
            self.create_query_engine()

        window_response = self.query_engine.query(msg)

        output = self.provide_response(msg, window_response)

        return output

# vector_store = VectorStore()
# conn_string = create_connection_string()
#
#
# documents = SimpleDirectoryReader("documents").load_data()
# print("Document ID:", documents[0].doc_id)
#
# for document in documents:
#     document.text = document.text.replace('\x00', '')
#
# vector_store.process_document(documents)
#
# vector_store.create_query_engine()
#
# new_docs = SimpleDirectoryReader("new_document").load_data()
# print("Document ID:", documents[0].doc_id)
# for document in new_docs:
#     document.text = document.text.replace('\x00', '')
#
# vector_store.process_document(new_docs)
# chat_input  = "Does NAS and HPO improve accuracy?"
# window_response = vector_store.query_engine.query(
#     chat_input
# )
#
# print(dir(window_response.response))
# print(type(window_response))
# pprint_response(window_response, show_source=True)
#
# vector_store.provide_response(chat_input, window_response)