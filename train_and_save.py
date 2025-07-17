from backend.server_enhanced import model

features, labels = model.load_and_process_data()
model.train_models(features, labels)
model.save_models()
