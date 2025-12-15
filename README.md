# MlNx

A comprehensive machine learning library in Elixir using Nx, following a 15-commit educational curriculum.

## Usage

### Running Interactive Demos

You can execute the interactive demos located in the `examples/` directory using `mix run`.

**Latest Demo (Learning Rate Scheduler):**
```bash
mix run examples/06_lr_scheduler_demo.exs
```

**Other Available Demos:**
```bash
mix run examples/05_batch_training_demo.exs
mix run examples/04_normalization_demo.exs
mix run examples/03_regularization_demo.exs
mix run examples/02_loss_functions_demo.exs
mix run examples/01_gradient_descent_demo.exs
mix run examples/00_linreg_demo.exs
```

### Running Tests

To verify that everything is working correctly, you can run the full test suite:
```bash
mix test
```

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `ml_nx` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ml_nx, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/ml_nx>.

