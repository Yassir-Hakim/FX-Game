import numpy as np
import matplotlib.pyplot as plt

class Trader:
    def __init__(self, pounds: float, dollars: float = 0.0):
        self.pounds = pounds
        self.dollars = dollars

    def sell_all_pounds_to_mm(self, offer_rate: float):

        """
        Returns the trade the trader wants to make:
        sell all pounds at the proposed $/£ rate.
        """

        dollars_requested = self.pounds * offer_rate
        return self.pounds, dollars_requested

    def receive_mm_trade(self, pounds_sold: float, dollars_received: float) -> None:
        self.pounds -= pounds_sold
        self.dollars += dollars_received

    def use_bdc_to_sell_all_pounds(self, true_rate: float, fee: float) -> None: #If MM rejects
        dollars_received = self.pounds * (1 - fee) * true_rate
        self.dollars += dollars_received
        self.pounds = 0.0


class MarketMaker:
    def __init__(self, pounds: float = 0.0, dollars: float = 0.0):
        self.pounds = pounds
        self.dollars = dollars

    def accepts_trade(self, offer_rate: float, true_rate: float) -> bool:
        """
        Trader is selling pounds to the MM.

        MM accepts if the trader's offered rate is below the true rate.
        Example:
        true rate = 1.30
        trader offers = 1.27
        MM buys pounds cheaply, so accepts.
        """

        return offer_rate < true_rate

    def complete_trade(self, trader: Trader, pounds_sold: float, dollars_paid: float):
        trader.receive_mm_trade(pounds_sold, dollars_paid)
        self.pounds += pounds_sold
        self.dollars -= dollars_paid

class BureauDeChange:
    def __init__(self, trader_fee: float = 0.02):
        self.trader_fee = trader_fee

    def exchange_for_trader(self, trader: Trader, true_rate: float):
        trader.use_bdc_to_sell_all_pounds(true_rate=true_rate, fee=self.trader_fee)


class GameResult:
    def __init__(self, true_rate: float, accepted_by_mm: bool, final_pounds: float, final_dollars: float):
        self.true_rate = true_rate
        self.accepted_by_mm = accepted_by_mm
        self.final_pounds = final_pounds
        self.final_dollars = final_dollars


class OneRoundGame:
    def __init__(
        self,
        initial_rate: float = 1.25,
        sigma: float = 0.05,
        initial_pounds: float = 50000,

    ):
        self.initial_rate = initial_rate
        self.sigma = sigma
        self.initial_pounds = initial_pounds

    def play(self, offer_rate: float):
        trader = Trader(pounds=self.initial_pounds)
        market_maker = MarketMaker()
        bdc = BureauDeChange()

        true_rate = np.random.normal(self.initial_rate, self.sigma)

        pounds_sold, dollars_requested = trader.sell_all_pounds_to_mm(offer_rate)

        accepted = market_maker.accepts_trade(offer_rate, true_rate)

        if accepted:
            market_maker.complete_trade(trader, pounds_sold, dollars_requested)
        else:
            bdc.exchange_for_trader(trader, true_rate)

        return GameResult(
            true_rate=true_rate,
            accepted_by_mm=accepted,
            final_pounds=trader.pounds,
            final_dollars=trader.dollars,
        )


def estimate_average_result(offer_rate: float, n_games: int):
    game = OneRoundGame()

    total_dollars = 0.0

    for _ in range(n_games):
        result = game.play(offer_rate)
        total_dollars += result.final_dollars

    return total_dollars / n_games


def plot_monte_carlo_results(
    min_rate: float = 1.15,
    max_rate: float = 1.40,
    step: float = 0.001,
    n_games: int = 50000,
):
    offer_rates = np.arange(min_rate, max_rate + step, step)

    average_dollars_list = []

    for offer_rate in offer_rates:
        average_dollars = estimate_average_result( offer_rate=offer_rate, n_games=n_games)
        average_dollars_list.append(average_dollars)

    best_index = np.argmax(average_dollars_list)
    best_offer_rate = offer_rates[best_index]
    best_average_dollars = average_dollars_list[best_index]

    print(f"Best offer rate: {best_offer_rate:.4f}")
    print(f"Best average dollars: ${best_average_dollars:,.2f}")

    plt.plot(offer_rates, average_dollars_list)
    plt.xlabel("Offer rate P ($ per £)")
    plt.ylabel("Average final dollars")
    plt.title("Monte Carlo estimate of expected dollars by offer rate")
    plt.axvline(best_offer_rate, linestyle="--", label=f"Best P ≈ {best_offer_rate:.4f}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_monte_carlo_results()