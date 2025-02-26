import numpy as np
import pandas as pd
from recbole.trainer import Trainer

class CustomTrainer(Trainer):
    # def save_recommendations(self, test_data, predictions, file_path='recommendations.csv'):
    #     """Save recommendations to a CSV file."""
    #     user_ids = test_data['user_id'].numpy()
    #     item_ids = test_data['item_id'].numpy()
    #     scores = predictions.numpy()
        
    #     df = pd.DataFrame({
    #         'user_id': user_ids,
    #         'item_id': item_ids,
    #         'score': scores
    #     })
    #     df.to_csv(self.checkpoint_dir+'/'+file_path, index=False)

    # def save_test_set(self, test_data, file_path='test_set.csv'):
    #     """Save the test set to a CSV file."""
    #     user_ids = test_data['user_id'].numpy()
    #     item_ids = test_data['item_id'].numpy()
        
    #     df = pd.DataFrame({
    #         'user_id': user_ids,
    #         'item_id': item_ids
    #     })
    #     df.to_csv(self.checkpoint_dir+'/'+file_path, index=False)

    # def evaluate(self, test_data, model, save_results=True):
    #     """Override the evaluate method to save recommendations and test set."""
    #     # Original evaluation logic
    #     predictions = model.predict(test_data)
        
    #     # Save recommendations and test set
    #     if save_results:
    #         self.save_recommendations(test_data, predictions, 'recommendations.csv')
    #         self.save_test_set(test_data, 'test_set.csv')
        
    #     # Return evaluation metrics
    #     return self._calculate_metrics(test_data, predictions)


    def save_recommendations(self, test_data, model, file_path='recommendations.csv'):
        """Save recommendations to a CSV file by iterating over the test data loader."""
        user_ids, item_ids, scores = [], [], []
        
        for batch in test_data:
            batch = batch.to(model.device)  # Move batch to the correct device
            batch_user_ids = batch['user_id'].cpu().numpy()
            batch_item_ids = batch['item_id'].cpu().numpy()
            batch_scores = model.full_sort_predict(batch).cpu().numpy()  # Get predictions
            
            user_ids.extend(batch_user_ids)
            item_ids.extend(batch_item_ids)
            scores.extend(batch_scores)
        
        df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'score': scores
        })
        df.to_csv(file_path, index=False)

    def save_test_set(self, test_data, file_path='test_set.csv'):
        """Save the test set to a CSV file by iterating over the test data loader."""
        user_ids, item_ids = [], []
        
        for batch in test_data:
            batch_user_ids = batch['user_id'].cpu().numpy()  # Move to CPU and convert to numpy
            batch_item_ids = batch['item_id'].cpu().numpy()  # Move to CPU and convert to numpy
            
            user_ids.extend(batch_user_ids)
            item_ids.extend(batch_item_ids)
        
        df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids
        })
        df.to_csv(file_path, index=False)

    def evaluate(self, test_data, model, save_results=True):
        """Override the evaluate method to save recommendations and test set."""
        # Original evaluation logic
        predictions = model.predict(test_data)
        
        # Save recommendations and test set
        if save_results:
            self.save_recommendations(test_data, model, 'recommendations.csv')
            self.save_test_set(test_data, 'test_set.csv')
        
        # Return evaluation metrics
        return self._calculate_metrics(test_data, predictions)


