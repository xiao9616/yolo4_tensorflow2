# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/5/14 上午10:50    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : Train.py         
# @Software: PyCharm
# ============================================

from abc import *


class Train(ABC):
    '''
    抽象基类（一个接口）：
    '''

    @classmethod
    @abstractmethod
    def get_model():
        '''

        Returns:返回一个model

        '''

    @classmethod
    @abstractmethod
    def get_loss():
        '''

        Returns:返回loss函数

        '''

    @classmethod
    @abstractmethod
    def get_optimizer():
        '''

        Returns:优化器

        '''
