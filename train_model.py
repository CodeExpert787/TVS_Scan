import os
from data_processor import DataProcessor
from model_trainer import ModelTrainer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Define directories
    data_dir = 'data/raw'  # Directory containing raw data files
    processed_dir = 'data/processed'  # Directory for processed data
    model_dir = 'models'  # Directory for trained models
    
    try:
        # Initialize data processor
        logger.info("Initializing data processor...")
        processor = DataProcessor(data_dir)
        
        # Load and process data
        logger.info("Loading and processing data...")
        train_data, test_data = processor.load_data()
        
        # Save processed data
        logger.info("Saving processed data...")
        processor.save_processed_data(train_data, test_data, processed_dir)
        
        # Preprocess data for training
        logger.info("Preprocessing data for training...")
        X_train, y_train = processor.preprocess_data(train_data)
        X_test, y_test = processor.preprocess_data(test_data)
        
        # Initialize and train model
        logger.info("Initializing model trainer...")
        trainer = ModelTrainer(model_dir)
        
        # Train model
        logger.info("Training model...")
        train_metrics = trainer.train(X_train, y_train)
        logger.info(f"Training metrics: {train_metrics}")
        
        # Evaluate model
        logger.info("Evaluating model...")
        test_metrics = trainer.evaluate(X_test, y_test)
        logger.info(f"Test metrics: {test_metrics}")
        
        # Save model
        logger.info("Saving trained model...")
        trainer.save_model()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 