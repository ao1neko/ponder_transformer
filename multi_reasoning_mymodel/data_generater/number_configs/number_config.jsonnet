{
  seed: 42,
  min_num: 0,
  max_num: 99,
  output_dir_path: "/work01/aoki0903/PonderNet/multihop_experiment/datas",

  contents: [
    "train",
    "valid",
    "test"
  ],

  split_rate: [
    8,
    1,
    1
  ],

  assert std.length(self.contents) == std.length(self.split_rate),  
}
