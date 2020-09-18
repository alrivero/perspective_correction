import csv

def line_of(x, point_1, point_2):
  m = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])

  return m * (x - point_1[0]) + point_1[1]

def read_lines(csv_dir, img_dim):
  lines = []
  with open(csv_dir) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
      point_1 = (int(row[0]), int(row[1]))
      point_2 = (int(row[2]), int(row[3]))

      # Use the point slope form to solve for x = 0 / width of the image.
      # This provides a solution because the focal plaane and building
      # are not parallel.
      edge_left = (0, int(line_of(0, point_1, point_2)))
      edge_right = (img_dim[1], int(line_of(img_dim[1], point_1, point_2)))
      lines.append((edge_left, edge_right))
  
  return lines

def draw_lines_on_img(img, lines, color, line_width):
  for (point_1, point_2) in lines:
    cv2.line(img, point_1, point_2, color, line_width)