import os

import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'D:/Program Files/Tesseract-OCR/tesseract.exe'

IMAGE_DIRECTORY = "bin/facesheets"
DEBUG = 0


def draw_box(image, x, y, w, h, colour_space=(0, 0, 255), border_size=2):
    """
    Draws a box around an object on an image
    :param image: image to draw the box
    :param x: top left corner x-coordinate
    :param y: top left corner y-coordinate
    :param w: width of the bounding box
    :param h: height of the bounding box
    :param colour_space: a tuple defining the BGR values for the bounding box border
    :param border_size: Size of bounding box border
    """
    cv2.rectangle(image, (x, y), (x + w, y + h), colour_space, border_size)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def show_image(image, window_name="untitled"):
    """
    Displays a given image in a named window (default name, 'Untitled')
    :param image: Image to be displayed
    :param window_name: Name of the window, default name 'Untitled'
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def name_parser(text):
    """
    Extracts first, middle, last name from a text
    :param text: text with name
    :return: first, middle, last name
    """
    first_name, middle_name, last_name = '', '', ''
    if "," in text:
        text = text
        last_name, remaining_part = text.split(",")
        name_parts = remaining_part.strip().split(" ")
        if len(name_parts) == 2:
            first_name, middle_name = name_parts
        elif len(name_parts) == 1:
            first_name = name_parts[0]
        else:
            first_name = last_name
            last_name = None
    else:
        name_parts = text.split(" ")
        if len(name_parts) == 3:
            first_name, middle_name, last_name = name_parts
        elif len(name_parts) == 2:
            first_name, middle_name = name_parts
        elif len(name_parts) == 1:
            first_name = name_parts[0]
    return first_name.strip(), middle_name.strip(), last_name.strip()


def extract_name(image, data):
    num_points = len(data["level"])
    target_box = None
    field_label = None
    for point in range(num_points):
        text = data["text"][point]
        x, y, w, h = data["left"][point], data["top"][point], data["width"][point], data["height"][point]
        if text.lower().startswith("name"):
            target_box = (x, y, w, h)
            field_label = text
            break
    candidate_boxes = []
    if target_box:
        for point in range(num_points):
            text = data["text"][point]
            x, y, w, h = data["left"][point], data["top"][point], data["width"][point], data["height"][point]
            if abs(y - target_box[1]) < 10 and text != field_label and 0 < x - target_box[0] < 500:
                candidate_boxes.append((text, (x, y, w, h)))
    candidate_boxes = sorted(candidate_boxes, key=lambda x: x[1][0])
    field_value = " ".join(x[0] for x in candidate_boxes)
    return name_parser(field_value)


def extract_visit_id(image, data):
    num_points = len(data["level"])
    target_box = None
    field_label = None
    for point in range(num_points):
        text = data["text"][point]
        x, y, w, h = data["left"][point], data["top"][point], data["width"][point], data["height"][point]
        if text.lower().startswith("visit"):
            target_box = (x, y, w, h)
            field_label = text
            break
    candidate_boxes = []
    if target_box:
        for point in range(num_points):
            text = data["text"][point]
            x, y, w, h = data["left"][point], data["top"][point], data["width"][point], data["height"][point]
            if abs(y - target_box[1]) < 10 and text != field_label and ":" not in text and 0 < x - target_box[0] < 100:
                candidate_boxes.append((text, (x, y, w, h)))
    candidate_boxes = sorted(candidate_boxes, key=lambda x: x[1][0])
    field_value = " ".join(x[0] for x in candidate_boxes)
    return field_value


def extract_mr_id(image, data):
    num_points = len(data["level"])
    target_box = None
    field_label = None
    for point in range(num_points):
        text = data["text"][point]
        x, y, w, h = data["left"][point], data["top"][point], data["width"][point], data["height"][point]
        if text.lower().startswith("mr"):
            target_box = (x, y, w, h)
            field_label = text
            break
    candidate_boxes = []
    if target_box:
        for point in range(num_points):
            text = data["text"][point]
            x, y, w, h = data["left"][point], data["top"][point], data["width"][point], data["height"][point]
            if abs(y - target_box[1]) < 10 and text != field_label and ":" not in text and 0 < x - target_box[0] < 100:
                candidate_boxes.append((text, (x, y, w, h)))
    candidate_boxes = sorted(candidate_boxes, key=lambda x: x[1][0])
    field_value = " ".join(x[0] for x in candidate_boxes)
    return field_value


def extract_field(label, field_width, data):
    num_points = len(data["level"])
    target_box = None
    field_label = None
    for point in range(num_points):
        text = data["text"][point]
        x, y, w, h = data["left"][point], data["top"][point], data["width"][point], data["height"][point]
        if text.lower().startswith(label.lower()):
            target_box = (x, y, w, h)
            field_label = text
            break
    candidate_boxes = []
    if target_box:
        for point in range(num_points):
            text = data["text"][point]
            x, y, w, h = data["left"][point], data["top"][point], data["width"][point], data["height"][point]
            if abs(y - target_box[1]) < 10 and text != field_label \
                    and ":" not in text and 0 < x - target_box[0] < field_width:
                candidate_boxes.append((text, (x, y, w, h)))
    candidate_boxes = sorted(candidate_boxes, key=lambda x: x[1][0])
    field_value = " ".join(x[0] for x in candidate_boxes)
    return field_value


def main():
    image_paths = filter(lambda x: x.endswith(".jpg"), os.listdir(IMAGE_DIRECTORY))
    for image_path in image_paths:
        image_path = os.path.join(IMAGE_DIRECTORY, image_path)
        image = cv2.imread(image_path)
        # show_image(image)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        # DEBUGGING
        if DEBUG:
            num_points = len(data["level"])
            for point in range(num_points):
                text = data["text"][point]
                x, y, w, h = data["left"][point], data["top"][point], data["width"][point], data["height"][point]
                if text:
                    draw_box(image, x, y, w, h)
        else:
            # Extract Name
            print("Name: ", extract_name(image, data))
            # Extract Visit ID
            print("Visit ID: ", extract_visit_id(image, data))
            # Extract MR ID
            print("MR ID: ", extract_mr_id(image, data))
            # Extract Field
            fields = [("Name", 500), ("Visit", 100), ("MR", 100), ("BirthDate", 200), ("Age", 100),
                      ("Gender", 150), ("SSN", 100)]
            for label, field_width in fields:
                print(f"{label}: ", extract_field(label, field_width, data))
        # Display Image
        # resized_image = image_resize(image, width=1000)
        # show_image(resized_image)
        break




if __name__ == "__main__":
    main()
