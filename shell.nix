let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  buildInputs = with pkgs; [
    python38
  ] ++ (with pkgs.python3.pkgs; [
    poetry
  ]);
}
