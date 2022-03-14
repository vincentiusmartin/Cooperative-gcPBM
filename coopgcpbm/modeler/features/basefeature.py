import abc

class BaseFeature:
    @abc.abstractmethod
    def __init__(self, params, traindf):
        """
        Base constructor for feasture

        Args:
            **kwargs: all features needed by the class
        """
        pass

    @abc.abstractmethod
    def get_feature(self,seqcolname="Sequence"):
        pass


    def set_attrs(self, params, default_params):
        for k, v in default_params.items():
            setattr(self, k, v)
        # we then replace the default with user param
        for k, v in params.items():
            if k not in default_params.keys():
                raise Exception("%s is not a valid parameter for %s" % (k,self.__class__.__name__))
            if k in default_params:
                # we only set if k is permitted (only those in default)
                setattr(self, k, v)
