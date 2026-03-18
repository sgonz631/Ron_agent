CREATE TABLE shoes (
  shoe_id INTEGER PRIMARY KEY AUTOINCREMENT,
  brand TEXT NOT NULL,
  model TEXT NOT NULL,
  size REAL NOT NULL,
  primary_color TEXT NOT NULL,
  cost REAL NOT NULL,
  quantity INTEGER NOT NULL CHECK (quantity >= 0),
  photo_url TEXT
);
