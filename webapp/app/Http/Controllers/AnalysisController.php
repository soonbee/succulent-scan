<?php

namespace App\Http\Controllers;

use App\Http\Requests\StoreAnalysisRequest;
use App\Models\Analysis;
use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\RedirectResponse;
use Illuminate\Support\Facades\Http;
use Inertia\Inertia;
use Inertia\Response;

class AnalysisController extends Controller
{
    public function store(StoreAnalysisRequest $request): Response|RedirectResponse
    {
        $imagePath = $request->file('image')->store('analyses', 'public');

        $analysis = Analysis::create([
            'image_path' => $imagePath,
            'status' => 'pending',
        ]);

        try {
            $response = Http::attach(
                'file',
                file_get_contents($request->file('image')->getRealPath()),
                $request->file('image')->getClientOriginalName()
            )->connectTimeout(5)->timeout(30)->post(config('services.inference.url').'/inference');

            if ($response->failed()) {
                $analysis->update(['status' => 'failed']);

                return redirect()->route('home');
            }

            $results = $response->json();

            $analysis->update([
                'status' => 'completed',
                'first_en' => $results[0]['en'],
                'first_acc' => $results[0]['acc'],
                'second_en' => $results[1]['en'],
                'second_acc' => $results[1]['acc'],
                'third_en' => $results[2]['en'],
                'third_acc' => $results[2]['acc'],
            ]);

            return Inertia::render('Result', [
                'results' => $results,
            ]);
        } catch (ConnectionException $e) {
            $analysis->update(['status' => 'failed']);

            $message = str_contains($e->getMessage(), 'Operation timed out')
                ? '분석 시간이 초과되었어요. 잠시 후 다시 시도해 주세요.'
                : '분석 서버에 연결할 수 없어요. 잠시 후 다시 시도해 주세요.';

            return redirect()->route('home');
        } catch (\Throwable) {
            $analysis->update(['status' => 'failed']);

            return redirect()->route('home');
        }
    }
}
