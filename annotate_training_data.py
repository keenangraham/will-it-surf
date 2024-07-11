from willitsurf import data


def annotate():
    data.annotate_raw_data(
        './assets/data/raw',
        './assets/data/labels/annotations.tsv',
    )


if __name__ == '__main__':
    annotate()
