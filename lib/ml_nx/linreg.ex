defmodule MLNx.LinReg do
  import Nx.Defn

  # Model: y_hat = Xw + b
  defn predict(x, w, b) do
    Nx.dot(x, w) + b
  end

  # Mean Squared Error Loss
  defn mse(y_hat, y) do
    err = y_hat - y
    Nx.mean(err * err)
  end

  defn loss(x, y, w, b) do
    y_hat = predict(x, w, b)
    mse(y_hat, y)
  end

  defn step(x, y, w, b, lr) do
      {gw, gb} = grad({w, b}, fn {w, b} ->
        loss(x, y, w, b)
      end)

      w = w - lr * gw
      b = b - lr * gb

      {w, b}
  end

  def train(x, y, opts \\ []) do
    iters = Keyword.get(opts, :iters, 500)
    lr = Keyword.get(opts, :lr, 0.05)

    {_, d} = Nx.shape(x)

    w = Nx.broadcast(0.0, {d})
    b = Nx.tensor(0.0)
    lr = Nx.tensor(lr)

    Enum.reduce(1..iters, {w, b}, fn _, {w, b} ->
      step(x, y, w, b, lr)
    end)

  end
end
