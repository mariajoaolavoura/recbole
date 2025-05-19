
import os
from logging import getLogger
from time import time
import datetime

import numpy as np
import torch
from tqdm import tqdm

from recbole.trainer import Trainer

# from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
# from recbole.evaluator import Evaluator, Collector

# from .generate_artificial_random_dataset import save_picklefile, validate_folderpath
from .file_utils import save_picklefile, validate_folderpath


from recbole.utils import (
    early_stopping,
    dict2str,
    EvaluatorType,
    set_color,
    get_gpu_usage
)

class CustomTrainer(Trainer):

    def __init__(self, config, model):
        super().__init__(config, model)

    ## modified version, with prints
    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        print('Entered Trainer fit function. Model is being trained.')

        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result



    ## modified version, with prints
    def _neg_sample_batch_eval(self, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        # print(type(batched_data), '<- IN CUSTOMTRAINER._neg_sample_batch_eval is batched_data a NegSampleEvalDataLoader or a Dataset?')
        # print(type(interaction), '<- IN CUSTOMTRAINER._neg_sample_batch_eval is interaction a NegSampleEvalDataLoader or a Dataset?')
        
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
            # print('IN CUSTOMTRAINER._neg_sample_batch_eval USED model.predict')
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)
            # print('IN CUSTOMTRAINER._neg_sample_batch_eval USED _spilt_predict')

        # print('IN CUSTOMTRAINER._neg_sample_batch_eval origin_scores', origin_scores)

        if self.config["eval_type"] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        elif self.config["eval_type"] == EvaluatorType.RANKING:
            col_idx = interaction[self.config["ITEM_ID_FIELD"]]
            batch_user_num = positive_u[-1] + 1
            # print('IN CUSTOMTRAINER._neg_sample_batch_eval batch_user_num=', batch_user_num, 'positive_u[-1]', positive_u[-1], 'positive_u size=', positive_u.size(), positive_u)
            # print('positive_i', positive_i,'size=', positive_i.size())
            scores = torch.full(
                (batch_user_num, self.tot_item_num), -np.inf, device=self.device
            )

            # print('IN CUSTOMTRAINER._neg_sample_batch_eval row_idx=', row_idx, '(', len(row_idx), ')')
            # print('IN CUSTOMTRAINER._neg_sample_batch_eval col_idx=interaction[self.config["ITEM_ID_FIELD"]]', col_idx, '(', len(col_idx), ')')
            # print('IN CUSTOMTRAINER._neg_sample_batch_eval interaction[self.config["USER_ID_FIELD"]]', interaction[self.config["USER_ID_FIELD"]], '(', len(interaction[self.config["USER_ID_FIELD"]]), ')')
            scores[row_idx, col_idx] = origin_scores
            return interaction, scores, positive_u, positive_i
        

    ## modified version, with prints
    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return        

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        # print(type(eval_data), '<- IN CUSTOMTRAINER.evaluate is eval_data a NegSampleEvalDataLoader or a Dataset?')
        # print(type(iter_data), '<- IN CUSTOMTRAINER.evaluate is iter_data a NegSampleEvalDataLoader or a Dataset?')

        # self.external_ids_items_rec = []
        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            self.all_external_ids_items_rec = []
            num_sample += len(batched_data)
            # print('IN CUSTOMTRAINER.evaluate batch_idx=', batch_idx)
            # print('IN CUSTOMTRAINER.evaluate batched_data=', type(batched_data))
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            # print('IN CUSTOMTRAINER.evaluate interaction=', interaction)
            # print('interaction=', interaction)
            # print('IN CUSTOMTRAINER.evaluate scores=', scores, 'size=', scores.size() ,'all -inf?', torch.all(scores == -np.inf))
            # print('IN CUSTOMTRAINER.evaluate positive_u=', positive_u,'positive_i=', positive_i)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )
            self._eval_batch_collect(scores, interaction, positive_u, positive_i, eval_data)


            try:
                pti = eval_data.dataset.dataset_name[eval_data.dataset.dataset_name.rindex('pt'):]
            except:
                pti = 'none'

            folder_path = self.config['checkpoint_dir']+'/'+'recommendations/'  
            rec_filename = folder_path+\
                            'RecExtItemIds_'+self.config['model']+\
                                                            '_test_'+pti+\
                                                                '_batch_'+str(batch_idx)+\
                                                                    '_'+str(round(time(), 4))+'.pkl'
                                                                    # '_'+str(datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'))+'.pkl'
             
             
            
            validate_folderpath(folder_path)
            save_picklefile(self.all_external_ids_items_rec, rec_filename)


        # pti = self.config['dataset'][self.config['dataset'].rindex('pt'):]
        # rec_filename = self.config['checkpoint_dir']+'/'+\
        #                 'RecExtItemIds_'+self.config['model']+\
        #                                                 '_test_'+pti+\
        #                                                         '_'+str(time())+'.pkl'
        # save_picklefile(self.external_ids_items_rec, rec_filename)


        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        # print('struct keys=',struct.keys())
        # print('IN CUSTOMTRAINER.evaluate struct.topk=',struct['rec.topk'])
        # print('IN CUSTOMTRAINER.evaluate size=',struct['rec.topk'].size())
        # torch.save(struct['rec.topk'], 'C:/Users/mjlav/Desktop/work/european_comission/recbole/notebooks/processed_datasets/natural_data/palco2010/more_2interQ_df_no_drift_pt1/trainer_struct.pt')
        # print('struct.items=',struct['rec.items'])
        # print('struct.label=',struct['rec.label'])
        
        result = self.evaluator.evaluate(struct)
        # print('result=',result)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result
    

    def _eval_batch_collect(self,
                            scores_tensor: torch.Tensor,
                            interaction,
                            positive_u: torch.Tensor,
                            positive_i: torch.Tensor,
                            eval_data):
        ''' copy of the evaluator.collector.Collector.eval_batch_collect() with save of the topk file'''

        if self.eval_collector.register.need("rec.topk"):
            # print('IN CUSTOMTRAINER.eval_batch_collect entered rec.topk')
            _, topk_idx = torch.topk(
                scores_tensor, max(self.eval_collector.topk), dim=-1
            )  # n_users x k

            # print('IN CUSTOMTRAINER._eval_batch_collect id2token from topk_idx\n',
            #       eval_data.dataset.id2token(eval_data.dataset.iid_field,
            #                                   topk_idx))
            self.all_external_ids_items_rec = eval_data.dataset\
                                                    .id2token(eval_data.dataset.iid_field,
                                                              topk_idx)
                                                              
            # print('IN CUSTOMTRAINER._eval_batch_collect self.all_external_ids_items_rec=', self.all_external_ids_items_rec)
            # print('is it yielding an error?', 
            #       eval_data._dataset.id2token(eval_data._dataset.iid_field,
            #                                   [-1]))
            # print('IN CUSTOMTRAINER.eval_batch_collect topk_idx=', topk_idx, '(', topk_idx.size(), ')')
            # pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
            # pos_matrix[positive_u, positive_i] = 1
            # print('IN CUSTOMTRAINER.eval_batch_collect pos_matrix=', pos_matrix, '(', pos_matrix.size(), ')', 'is all 0?', torch.all(pos_matrix == 0))
            # pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            # print('IN CUSTOMTRAINER.eval_batch_collect pos_len_list=', pos_len_list, '(', pos_len_list.size(), ')')
            # pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            # print('IN CUSTOMTRAINER.eval_batch_collect pos_idx=', pos_idx, '(', pos_idx.size(), ')')
            # result = torch.cat((pos_idx, pos_len_list), dim=1)
            # print('IN CUSTOMTRAINER.eval_batch_collect torch.cat((pos_idx, pos_len_list), dim=1)=result=', result, '(', result.size(), ')')
            # self.eval_collector.data_struct.update_tensor("rec.topk", result)