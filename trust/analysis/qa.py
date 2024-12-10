# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import requests
import pandas as pd
from tqdm import tqdm

import trust as tr

logger = tr.CustomLogger(__name__)  # use custom logger


class QA:

    def __init__(self,
                 file_cheaters: str,
                 job_id: int):
        # csv file with cheaters
        self.file_cheaters = file_cheaters
        # appen job ID
        self.job_id = job_id

    def ban_users(self):
        """
        Ban users described in csv file self.file_cheaters from job
        self.job_id.
        """
        # import csv file
        df = pd.read_csv(self.file_cheaters)
        # check if there are users to flag
        if df.shape[0] == 0:
            return
        logger.info('Banning {} users.', df.shape[0])
        # count banned users
        banned_counter = 0
        # loop over users in the job for flagging
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # check if worker ID is not nan
            if pd.isna(row['worker_id']):
                logger.debug('User with NaN as worker ID detected in the list '
                             + 'of cheaters.')
                continue
            # make a PUT request for flagging
            cmd_put = 'https://api.appen.com/v1/jobs/' + \
                      str(self.job_id) + \
                      '/workers/' + \
                      str(int(row['worker_id'])) + \
                      '/ban.json'
            if not pd.isna(row['worker_code']):
                reason_text = 'User repeatedly ignored our ' \
                            + 'instructions and joined job from different ' \
                            + 'accounts/IP addresses. The same code ' \
                            + str(row['worker_code']) \
                            + ' used internally in the job was reused.'
            else:
                reason_text = 'User repeatedly ignored our  ' \
                            + 'instructions and joined job from different  ' \
                            + 'accounts/IP addresses. No worker code used  ' \
                            + 'internally was inputted (html regex ' \
                            + 'validator was bypassed).'
            params = {'reason': reason_text,
                      'key': tr.common.get_secrets('appen_api_key')}
            headers = {'Authorization': 'Token token=' + tr.common.get_secrets('appen_api_key')}
            # send PUT request
            try:
                # TODO: This API call seems to be broken and it returns 401.
                # working cmd curl -X PUT "https://api.appen.com/v1/jobs/2190951/workers/47426294/ban.json" -H "Authorization: Token token=APIKEY" -d "reason=Pavlo Bazilinskyy"  # noqa: E501
                r = requests.put(cmd_put, data=params, headers=headers)
            except requests.exceptions.ConnectionError:
                logger.error('No internet connection. Could not flag user {}.',
                             str(row['worker_id']))
                continue
            # code 200 means success
            code = r.status_code
            msg = r.content.decode()
            if (code == 200
               and msg != 'Contributor has already been banned'):
                banned_counter += 1
            logger.debug('Banned user {} with message \'{}\'. Returned '
                         + 'code {}: {}',
                         str(row['worker_id']),
                         reason_text,
                         str(code),
                         r.content)
        logger.info('Banned {} users successfully (users not banned '
                    + 'previously).',
                    str(banned_counter))

    def reject_users(self):
        """
        Reject users described in csv file self.file_cheaters from job
        self.job_id.
        """
        # import csv file
        df = pd.read_csv(self.file_cheaters)
        # check if there are users to reject
        if df.shape[0] == 0:
            return
        logger.info('Rejecting {} users.', df.shape[0])
        # count rejected users
        rejected_counter = 0
        # loop over users in the job for rejecting
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # check if worker ID is not nan
            if pd.isna(row['worker_id']):
                logger.debug('User with NaN as worker ID detected in the list '
                             + 'of cheaters.')
                continue
            # make a PUT request for rejecting
            cmd_put = 'https://api.appen.com/v1/jobs/' + \
                      str(self.job_id) + \
                      '/workers/' + \
                      str(int(row['worker_id'])) + \
                      '/reject.json'
            if not pd.isna(row['worker_code']):
                reason_text = 'User repeatedly ignored our instructions and ' \
                            + 'joined job from different accounts/IP ' \
                            + 'addresses. The same code ' \
                            + str(row['worker_code']) \
                            + ' used internally in the job was reused.'
            else:
                reason_text = 'User repeatedly ignored our instructions and ' \
                            + 'joined job from different accounts/IP ' \
                            + 'addresses. No worker code used internally  ' \
                            + 'was inputted (html regex validator was ' \
                            + 'bypassed).'
            params = {'reason': reason_text,
                      'manual': 'true',
                      'key': tr.common.get_secrets('appen_api_key')}
            headers = {'Authorization': 'Token token=' + tr.common.get_secrets('appen_api_key')}

            # send PUT request
            try:
                r = requests.put(cmd_put, data=params, headers=headers)
            except requests.exceptions.ConnectionError:
                logger.error('No internet connection. Could not reject user ' +
                             '{}.', str(row['worker_id']))
                continue
            # code 200 means success
            code = r.status_code
            msg = r.content.decode()
            if code == 200:
                rejected_counter += 1
            logger.debug('Rejected user {} with message \'{}\'. Returned '
                         + 'code {}: {}',
                         str(row['worker_id']),
                         reason_text,
                         str(code),
                         msg)
        logger.info('Rejected {} users successfully (users not rejected '
                    + 'previously).',
                    str(rejected_counter))
