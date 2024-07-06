from willitsurf import feed, data


def collect():
    data.collect_training_example(
        './assets/data/raw',
        feed.urls
    )


if __name__ == '__main__':
    collect()
