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
            )->timeout(10)->post(config('services.inference.url').'/inference');

            if ($response->failed()) {
                $analysis->update(['status' => 'failed']);

                return redirect()->route('home')->with('error', [
                    'title' => '분석을 완료하지 못했어요',
                    'description' => '다시 한 번 시도해 보시고, 같은 문제가 반복되면 관리자에게 문의해 주세요.',
                ]);
            }

            $data = $response->json();
            $reliable = $data['reliable'];
            $results = $data['results'];

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
                'reliable' => $reliable,
                'results' => $results,
            ]);
        } catch (ConnectionException $e) {
            $analysis->update(['status' => 'failed']);

            $error = str_contains($e->getMessage(), 'Operation timed out')
                ? [
                    'title' => '잠시 요청이 몰리고 있어요',
                    'description' => '현재 요청이 많아 분석을 바로 처리하지 못했어요. 조금만 기다렸다가 다시 시도해 주세요.',
                ]
                : [
                    'title' => '분석 서버에 연결할 수 없어요',
                    'description' => '잠시 후 다시 시도해 보시고, 문제가 계속되면 관리자에게 문의해 주세요.',
                ];

            return redirect()->route('home')->with('error', $error);
        } catch (\Throwable) {
            $analysis->update(['status' => 'failed']);

            return redirect()->route('home')->with('error', [
                'title' => '분석을 완료하지 못했어요',
                'description' => '다시 한 번 시도해 보시고, 같은 문제가 반복되면 관리자에게 문의해 주세요.',
            ]);
        }
    }
}
