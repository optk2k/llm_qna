import pathlib
from typing import List

import kagglehub # type: ignore
from langchain_core.documents.base import Document
import typer
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import Annotated

app = typer.Typer()
embeddings = OllamaEmbeddings(model="llama3")


@app.command()
def data_to_vector() -> None:
    path_csv = kagglehub.dataset_download(
        "shohinurpervezshohan/freelancer-earnings-and-job-trends", force_download=True
    )
    loader = CSVLoader(
        file_path=pathlib.Path(path_csv)
        .joinpath("freelancer_earnings_bd.csv")
        .__str__()
    )
    docs: List[Document] = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits: List[Document] = text_splitter.split_documents(docs)
    document_ids: FAISS = FAISS.from_documents(all_splits, embeddings)
    document_ids.save_local("faiss_index")


@app.command()
def question(
    request: str,
    model: Annotated[str, typer.Option("--model", "-m")] = "llama3",
    temperature: Annotated[float, typer.Option("--temperature", "-t")] = 0.1,
) -> None:
    document_ids: FAISS = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        num_predict=256,
    )
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=document_ids.as_retriever())
    response = chain.invoke(request)
    print("Ответ:", response["result"])


if __name__ == "__main__":
    app()
