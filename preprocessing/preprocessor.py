class Preprocessor:
    """
    Preprocessor Pipeline
    """
    filter_list = []

    def add_filter(self, filter):
        if filter is not None:
            self.filter_list.append(filter)

    def process(self, img):

        for filter in self.filter_list:
            if filter is not None:
                img= filter(img)


        return img


