import pandas as pd
import numpy as np
from os import path


class DMMetaManager(object):
    '''Class for reading meta data and feeding them to training
    '''

    def __init__(self, 
                 img_tsv='./metadata/images_crosswalk.tsv', 
                 exam_tsv='./metadata/exams_metadata.tsv', 
                 img_folder='./trainingData', 
                 img_extension='dcm'):
        '''Constructor for DMMetaManager
        Args:
            img_tsv ([str]): path to the image meta .tsv file. 
            exam_tsv ([str]): path to the exam meta .tsv file. Default is None
                    because this file is not available to SC1. 
            img_folder ([str]): path to the folder where the images are stored.
            img_extension ([str]): image file extension. Default is 'dcm'.
        '''

        def mod_file_path(name):
            '''Change file name extension and append folder path.
            '''
            return path.join(img_folder, 
                             path.splitext(name)[0] + '.' + img_extension)

        img_df = pd.read_csv(img_tsv, sep="\t", na_values=['.', '*'])
        try:
            img_df_indexed = img_df.set_index(['subjectId', 'examIndex'])
        except KeyError:
            img_df_indexed = img_df.set_index(['subjectId'])
        if exam_tsv != "":
            exam_df = pd.read_csv(exam_tsv, sep="\t", na_values=['.', '*'])
            exam_df_indexed = exam_df.set_index(['subjectId', 'examIndex'])
            self.exam_img_df = exam_df_indexed.join(img_df_indexed)
            self.exam_img_df['filename'] = \
                self.exam_img_df['filename'].apply(mod_file_path)
        else:
            img_df_indexed['filename'] = \
                img_df_indexed['filename'].apply(mod_file_path)
            self.img_df_indexed = img_df_indexed

        # Setup CC and MLO view categorization.
        # View      Description
        # *Undetermined yet.
        # AT        axillary tail  *
        # CC        craniocaudal
        # CCID      craniocaudal (implant displaced)
        # CV        cleavage  *
        # FB        from below
        # LM        90 lateromedial
        # LMO       lateromedial oblique
        # ML        90 mediolateral
        # MLID      90 mediolateral (implant displaced)
        # MLO       mediolateral oblique
        # MLOID     mediolateral oblique (implant displaced)
        # RL        rolled lateral  *
        # RM        rolled medial  *
        # SIO       superior inferior oblique
        # XCCL      exaggerated craniocaudal lateral
        # XCCM      exaggerated craniocaudal medial
        self.view_cat_dict = {
            'CC': 'CC', 'CCID': 'CC', 'FB': 'CC', 'LM': 'CC', 
            'ML': 'CC', 'MLID': 'CC', 'XCCL': 'CC', 'XCCM': 'CC',
            'MLO': 'MLO', 'LMO': 'MLO', 'MLOID': 'MLO', 'SIO': 'MLO'}


    def get_exam_df(self):
        '''Get exam dataframe
        '''
        try:
            return self.exam_img_df
        except AttributeError:
            return self.img_df_indexed


    def get_flatten_img_list(self, meta=False):
        '''Get image-level training data list
        Args:
            meta ([bool]): whether to return meta info or not. Default is 
                    False.
        '''
        img = []
        lab = []
        try:
            for idx, dat in self.exam_img_df.iterrows():
                img_name = dat['filename']
                laterality = dat['laterality']
                cancer = dat['cancerL'] if laterality == 'L' else dat['cancerR']
                try:
                    cancer = int(cancer)
                except ValueError:
                    cancer = np.nan
                img.append(img_name)
                lab.append(cancer)
        except AttributeError:
            for idx, dat in self.img_df_indexed.iterrows():
                img_name = dat['filename']
                try:
                    cancer = int(dat['cancer'])
                except KeyError:
                    cancer = np.nan
                img.append(img_name)
                lab.append(cancer)

        return (img, lab)


    def _get_info_per_exam(self, exam):
        '''Get training-related info for each exam as a dict
        Args:
            exam (DataFrame): data for an exam.
        Returns:
            A dict containing info for each breast: the cancer status for 
            each breast and the image paths to the CC and MLO views. If an 
            image is missing, the corresponding path is None. 
        Notes:
            In current implementation, only CC and MLO views are included. 
            All other meta info are not included.
        '''

        info = {'L': {}, 'R': {}}
        exam_indexed = exam.set_index(['laterality', 'view', 'imageIndex'])
        # Determine cancer status.
        try:
            cancerL = exam_indexed['cancerL'].iloc[0]
            cancerR = exam_indexed['cancerR'].iloc[0]
            try:
                cancerL = int(cancerL)
            except ValueError:
                cancerL = np.nan
            try:
                cancerR = int(cancerR)
            except ValueError:
                cancerR = np.nan
        except KeyError:
            try:
                cancerL = int(exam_indexed.loc['L']['cancer'].iloc[0])
            except KeyError:
                cancerL = np.nan
            try:
                cancerR = int(exam_indexed.loc['R']['cancer'].iloc[0])
            except KeyError:
                cancerR = np.nan
        info['L']['cancer'] = cancerL
        info['R']['cancer'] = cancerR
        info['L']['CC'] = None
        info['R']['CC'] = None
        info['L']['MLO'] = None
        info['R']['MLO'] = None
        for breast in exam_indexed.index.levels[0]:
            for view in exam_indexed.loc[breast].index.levels[0]:
                if view not in self.view_cat_dict:
                    continue  # skip uncategorized view for now.
                view_ = self.view_cat_dict[view]
                fname_df = exam_indexed.loc[breast].loc[view][['filename']]
                if fname_df.empty:
                    continue
                if info[breast][view_] is None:
                    info[breast][view_] = fname_df
                elif view == 'CC' or view == 'MLO':
                    # Make sure canonical views are always on top.
                    info[breast][view_] = fname_df.append(info[breast][view_])
                else:
                    info[breast][view_] = info[breast][view_].append(fname_df)

        return info


    def subj_generator(self):
        '''A generator for the data of each subject
        Returns:
            A tuple of (subject ID, the corresponding records of the subject).
        '''
        try:
            df = self.exam_img_df
        except AttributeError:
            df = self.img_df_indexed
        try:
            subj_list = df.index.levels[0]
        except AttributeError:
            subj_list = df.index.unique()
        for subj_id in subj_list:
            yield (subj_id, df.loc[subj_id])


    def exam_generator(self):
        '''A generator for the data of each exam
        Returns:
            A tuple of (subject ID, exam Index, the corresponding records of 
            the exam).
        Notes:
            All exams are flattened. When examIndex is unavailable, the 
            returned exam index is equal to the subject ID.
        '''
        for subj_id, subj_dat in self.subj_generator():
            for ex_idx in subj_dat.index.unique():
                yield (subj_id, ex_idx, subj_dat.loc[ex_idx])


    def last_exam_generator(self):
        '''A generator for the data of the last exam of each subject
        Returns:
            A tuple of (subject ID, exam Index, the corresponding records of 
            the exam).
        Notes:
            When examIndex is unavailable, the returned exam index is equal to 
            the subject ID.
        '''
        for subj_id, subj_dat in self.subj_generator():
            last_idx = subj_dat.index.max()
            yield (subj_id, last_idx, subj_dat.loc[last_idx])


    def last_2_exam_generator(self):
        '''A generator for the data of the last 2 exams of each subject
        Returns:
            A tuple of (subject ID, current exam Index, current exam data,
            prior exam Index, prior exam data). If no prior exam is present, 
            will return None.
        Notes:
            for SC2.
        '''
        for subj_id, subj_dat in self.subj_generator():
            nb_exam = len(subj_dat.index.unique())
            if nb_exam == 1:
                yield (subj_id, 1, subj_dat.loc[1], None, None)
            for prior_idx in xrange(1, nb_exam):
                curr_idx = prior_idx + 1
                yield (subj_id, curr_idx, subj_dat.loc[curr_idx], 
                       prior_idx, subj_dat.loc[prior_idx])


    def get_last_2_exam_dat(self):
        '''Get the info about the last 2 exams as a dataframe
        Returns: 
            a tuple of (df, labs) where df is a dataframe of exam pair info 
            for breasts; labs is the corresponding cancer labels.
        '''
        rec_list = []
        lab_list = []
        for subj_id, curr_idx, curr_dat, prior_idx, prior_dat in \
                self.last_2_exam_generator():
            left_record, right_record = \
                DMMetaManager.get_info_exam_pair(curr_dat, prior_dat)
            rec_list.append(left_record)
            rec_list.append(right_record)

            try:
                left_cancer = int(curr_dat['cancerL'].iloc[0])
            except ValueError:
                left_cancer = np.nan
            try:
                right_cancer = int(curr_dat['cancerR'].iloc[0])
            except ValueError:
                right_cancer = np.nan
            lab_list.append(left_cancer)
            lab_list.append(right_cancer)

        df = pd.concat(rec_list, ignore_index=True)
        labs = np.array(lab_list)
        
        return df, labs


    def get_subj_list(self, meta=False):
        '''Get subject-level training data list
        Returns:
            A list of all subjects. Each element is a tuple of (subject ID, 
            [ (exam Index, extracted exam info), ..., () ] ).
        '''
        subj_list = []
        for subj_id, subj_dat in self.subj_generator():
            subj_exam_list = []
            for ex_idx in subj_dat.index.unique():  # uniq exam indices.
                exam_info = self._get_info_per_exam(subj_dat.loc[ex_idx])
                subj_exam_list.append( (ex_idx, exam_info) )
            subj_list.append( (subj_id, subj_exam_list) )
        return subj_list


    def get_flatten_exam_list(self, meta=False):
        '''Get exam-level training data list
        Returns:
            A list of all exams for all subjects. Each element is a tuple of 
            (subject ID, exam Index, a dict of extracted info for the exam).
        '''
        exam_list = []
        for subj_id, ex_idx, exam_dat in self.exam_generator():
            exam_list.append( (subj_id, ex_idx, 
                               self._get_info_per_exam(exam_dat)) )
        return exam_list


    @staticmethod
    def get_info_exam_pair(curr_dat, prior_dat):
        '''Extract meta info from current and prior exams
        Returns: 
            a tuple of (left_df, right_df), where left_df and right_df are both
            dataframes containing meta info about the current and prior exams.
        '''
        # number of days since last exam.
        nb_days = curr_dat['daysSincePreviousExam'].iloc[0]
        # prior cancer invasive or not.
        try:
            left_prior_inv = prior_dat['invL'].iloc[0]
        except TypeError:
            left_prior_inv = np.nan
        try:
            right_prior_inv = prior_dat['invR'].iloc[0]
        except TypeError:
            right_prior_inv = np.nan
        # current, prior and diff bmi.
        curr_bmi = curr_dat['bmi'].iloc[0]
        try:
            prior_bmi = prior_dat['bmi'].iloc[0]
            diff_bmi = (curr_bmi - prior_bmi)/nb_days*365
        except TypeError:
            prior_bmi = np.nan
            diff_bmi = np.nan
        # implantation.
        implantNow = curr_dat['implantNow'].iloc[0]
        if implantNow == 2:
            left_implantNow = 1
            right_implantNow = 0
        elif implantNow == 1:
            left_implantNow = 0
            right_implantNow = 1
        elif implantNow == 4:
            left_implantNow = 1
            right_implantNow = 1            
        elif implantNow == 5:
            left_implantNow = .5
            right_implantNow = .5
        else:
            left_implantNow = np.nan
            right_implantNow = np.nan
        try:
            implantPrior = prior_dat['implantNow'].iloc[0]
            if implantPrior == 2:
                left_implantPrior = 1
                right_implantPrior = 0
            elif implantPrior == 1:
                left_implantPrior = 0
                right_implantPrior = 1
            elif implantPrior == 4:
                left_implantPrior = 1
                right_implantPrior = 1
            elif implantPrior == 5:
                left_implantPrior = .5
                right_implantPrior = .5
            else:
                left_implantPrior = np.nan
                right_implantPrior = np.nan
        except TypeError:
            left_implantPrior = np.nan
            right_implantPrior = np.nan                
        # previous breast cancer history.
        previousBcLaterality = curr_dat['previousBcLaterality'].iloc[0]
        if previousBcLaterality == 2:
            left_previousBcHistory = 1
            right_previousBcHistory = 0
        elif previousBcLaterality == 1:
            left_previousBcHistory = 0
            right_previousBcHistory = 1
        elif previousBcLaterality == 3:
            left_previousBcHistory = .5
            right_previousBcHistory = .5
        elif previousBcLaterality == 4:
            left_previousBcHistory = 1
            right_previousBcHistory = 1
        else:
            left_previousBcHistory = 0
            right_previousBcHistory = 0
        # breast reduction history.
        reduxLaterality = curr_dat['reduxLaterality'].iloc[0]
        if reduxLaterality == 2:
            left_reduxHistory = 1
            right_reduxHistory = 0
        elif reduxLaterality == 1:
            left_reduxHistory = 0
            right_reduxHistory = 1
        elif reduxLaterality == 4:
            left_reduxHistory = 1
            right_reduxHistory = 1
        else:
            left_reduxHistory = 0
            right_reduxHistory = 0
        # hormone replacement therapy.
        curr_hrt = curr_dat['hrt'].iloc[0]
        curr_hrt = np.nan if curr_hrt == 9 else curr_hrt
        try:
            prior_hrt = prior_dat['hrt'].iloc[0]
            prior_hrt = np.nan if prior_hrt == 9 else prior_hrt
        except TypeError:
            prior_hrt = np.nan
        # anti-estrogen therapy.
        curr_antiestrogen = curr_dat['antiestrogen'].iloc[0]
        curr_antiestrogen = np.nan if curr_antiestrogen == 9 else curr_antiestrogen
        try:
            prior_antiestrogen = prior_dat['antiestrogen'].iloc[0]
            prior_antiestrogen = np.nan if prior_antiestrogen == 9 else prior_antiestrogen
        except TypeError:
            prior_antiestrogen = np.nan
        # first degree relative with BC.
        firstDegreeWithBc = curr_dat['firstDegreeWithBc'].iloc[0]
        firstDegreeWithBc = np.nan if firstDegreeWithBc == 9 else firstDegreeWithBc
        # first degree relative with BC under 50.
        firstDegreeWithBc50 = curr_dat['firstDegreeWithBc50'].iloc[0]
        firstDegreeWithBc50 = np.nan if firstDegreeWithBc50 == 9 else firstDegreeWithBc50
        # race.
        race = curr_dat['race'].iloc[0]
        race = np.nan if race == 9 else race

        # put all info input a dict.
        left_record = {
            'daysSincePreviousExam': nb_days,
            'prior_inv': left_prior_inv,
            'age': curr_dat['age'].iloc[0],
            'implantEver': curr_dat['implantEver'].iloc[0],
            'implantNow': left_implantNow,
            'implantPrior': left_implantPrior,
            'previousBcHistory': left_previousBcHistory,
            'yearsSincePreviousBc': curr_dat['yearsSincePreviousBc'].iloc[0],
            'reduxHistory': left_reduxHistory,
            'curr_hrt': curr_hrt,
            'prior_hrt': prior_hrt,
            'curr_antiestrogen': curr_antiestrogen,
            'prior_antiestrogen': prior_antiestrogen,
            'curr_bmi': curr_bmi,
            'prior_bmi': prior_bmi,
            'diff_bmi': diff_bmi,
            'firstDegreeWithBc': firstDegreeWithBc,
            'firstDegreeWithBc50': firstDegreeWithBc50,
            'race': race
        }
        right_record = {
            'daysSincePreviousExam': nb_days,
            'prior_inv': right_prior_inv,
            'age': curr_dat['age'].iloc[0],
            'implantEver': curr_dat['implantEver'].iloc[0],
            'implantNow': right_implantNow,
            'implantPrior': right_implantPrior,
            'previousBcHistory': right_previousBcHistory,
            'yearsSincePreviousBc': curr_dat['yearsSincePreviousBc'].iloc[0],
            'reduxHistory': right_reduxHistory,
            'curr_hrt': curr_hrt,
            'prior_hrt': prior_hrt,
            'curr_antiestrogen': curr_antiestrogen,
            'prior_antiestrogen': prior_antiestrogen,
            'curr_bmi': curr_bmi,
            'prior_bmi': prior_bmi,
            'diff_bmi': diff_bmi,
            'firstDegreeWithBc': firstDegreeWithBc,
            'firstDegreeWithBc50': firstDegreeWithBc50,
            'race': race
        }
        return (pd.DataFrame(left_record, index=[0]), 
                pd.DataFrame(right_record, index=[0]))


    @staticmethod
    def exam_labs(exam_list):
        return [ 1 if e[2]['L']['cancer'] or e[2]['R']['cancer'] else 0 
                 for e in exam_list ]

    @staticmethod
    def flatten_exam_labs(exam_list):
        labs = []
        for e in exam_list:
            lc = e[2]['L']['cancer']
            rc = e[2]['R']['cancer']
            lc = lc if not np.isnan(lc) else 0
            rc = rc if not np.isnan(rc) else 0
            labs.append(lc)
            labs.append(rc)
        return labs

    @staticmethod
    def exam_list_summary(exam_list):
        '''Return a summary dataframe for an exam list
        '''
        subj_list = []
        exid_list = []
        l_cc_list = []
        l_mlo_list = []
        r_cc_list = []
        r_mlo_list = []
        l_can_list = []
        r_can_list = []
        def nb_fname(df):
            return 0 if df is None else df.shape[0]
        for e in exam_list:
            subj_list.append(e[0])
            exid_list.append(e[1])
            l_cc_list.append(nb_fname(e[2]['L']['CC']))
            l_mlo_list.append(nb_fname(e[2]['L']['MLO']))
            r_cc_list.append(nb_fname(e[2]['R']['CC']))
            r_mlo_list.append(nb_fname(e[2]['R']['MLO']))
            l_can_list.append(e[2]['L']['cancer'])
            r_can_list.append(e[2]['R']['cancer'])
        summary_df = pd.DataFrame(
            {'subj': subj_list, 'exam': exid_list, 
             'L_CC': l_cc_list, 'L_MLO': l_mlo_list,
             'R_CC': r_cc_list, 'R_MLO': r_mlo_list,
             'L_cancer': l_can_list, 'R_cancer': r_can_list})
        return summary_df


    def get_last_exam_list(self, meta=False):
        '''Get the last exam training data list
        Returns:
            A list of the last exams for each subject. Each element is a tuple 
            of (subject ID, exam Index, a dict of extracted info for the exam).
        '''
        exam_list = []
        for subj_id, ex_idx, exam_dat in self.last_exam_generator():
            exam_list.append( (subj_id, ex_idx, 
                               self._get_info_per_exam(exam_dat)) )
        return exam_list












