def add_predictions(packets, frame_numbers, predictor):
    """
    Add the prediction for a frame
    :param packets: list of scapy packets
    :param frame_numbers: frame numbers for prediction
    :param predictor: prediction value
    """
    for frame_num in frame_numbers:
        pkt = packets[frame_num - 1]
        pkt.prediction = predictor


def get_frame_numbers(connections):
    return [frm for frm, *_ in connections]
