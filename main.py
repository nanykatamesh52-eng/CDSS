from models.model import QAModel

if __name__ == "__main__":
    model = QAModel()
    file_path = 'data/CHI Drug Formulary (03Mar2024).xlsx'
    retriever = model.load_database(file_path)
    query = "Suggest me dosage for ACTIVE ENTHESITIS JUVENILE ARTHRITIS"
    response = model(query, retriever)
    print(response)
    